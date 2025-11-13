import os
import io
import tempfile

import numpy as np
import xarray as xr
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    after_this_request,
)
from stl import mesh

app = Flask(__name__)


# --- Utility functions -------------------------------------------------
def process_nc(file_storage, max_n=800):
    """
    アップロードされた nc ファイル(FileStorage)を一時ファイルに保存し、
    xarray で読み込んで間引きした lon/lat/Z を返す。
    """

    # 一時ファイルに保存
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    file_storage.save(tmp.name)
    tmp.close()

    try:
        ds = xr.open_dataset(tmp.name)

        # 標高データの変数名を推定
        cand_vars = ["elevation", "z", "bathymetry", "bedrock"]
        varname = None
        for v in cand_vars:
            if v in ds.data_vars:
                varname = v
                break
        if varname is None:
            if len(ds.data_vars) == 1:
                varname = list(ds.data_vars)[0]
            else:
                raise ValueError("標高データが特定できません。")

        da = ds[varname]

        # lon / lat 次元名
        lon_name = [c for c in da.dims if "lon" in c.lower() or c.lower() == "x"][0]
        lat_name = [c for c in da.dims if "lat" in c.lower() or c.lower() == "y"][0]

        lons = ds[lon_name].values
        lats = ds[lat_name].values
        Z = da.values

    finally:
        # 一時ファイル削除
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    # 行列が大きすぎると重いので間引き
    max_n = int(max_n)
    step_y = max(1, int(np.ceil(Z.shape[0] / max_n)))
    step_x = max(1, int(np.ceil(Z.shape[1] / max_n)))

    Z_sub = Z[::step_y, ::step_x]
    lats_sub = lats[::step_y]
    lons_sub = lons[::step_x]

    return lons_sub.tolist(), lats_sub.tolist(), Z_sub.tolist()


def build_solid_stl(lons, lats, Z, z_scale_visual):
    """
    lon/lat/Z から土台付き STL を作る関数（例）。
    関数のシグネチャはあなたの app.py に合わせてください。
    """

    # --- ここから上は今のコードに合わせて OK ---

    rows, cols = Z.shape

    # ① まず Z をスケーリング（STL 用に 10 倍など）
    z_scale_for_stl = float(z_scale_visual) * 10.0
    Z_scaled = Z * z_scale_for_stl

    # ② 土台の設定：高低差を元に厚さを決める
    relief = Z_scaled.max() - Z_scaled.min()
    base_thickness = max(relief * 0.1, 500.0)   # ← ここが新ロジック
    min_z_surface = Z_scaled.min()
    bottom_z_level = min_z_surface - base_thickness

    # ③ 頂点配列を用意
    num_top_vertices = rows * cols
    total_vertices = num_top_vertices * 2
    vertices = np.zeros((total_vertices, 3), dtype=np.float64)

    # ④ 表面頂点（ここで Z_scaled を使う）
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            x = (lons[j] - lon_origin) * meters_per_degree_lon
            y = (lats[i] - lat_origin) * meters_per_degree_lat
            z = Z_scaled[i, j]
            vertices[idx] = [x, y, z]

    # ⑤ 底面頂点（bottom_z_level を使用）
    for i in range(rows):
        for j in range(cols):
            idx_top = i * cols + j
            idx_bottom = num_top_vertices + idx_top
            vertices[idx_bottom] = [
                vertices[idx_top, 0],
                vertices[idx_top, 1],
                bottom_z_level,
            ]



    def top(i, j):
        return i * cols + j

    def bottom(i, j):
        return num_top + i * cols + j

    faces = []

    # 表面
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0, v1, v2, v3 = top(i, j), top(i, j + 1), top(i + 1, j), top(i + 1, j + 1)
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    # 側面（上端）
    for j in range(cols - 1):
        faces.append([top(0, j), bottom(0, j), bottom(0, j + 1)])
        faces.append([top(0, j), bottom(0, j + 1), top(0, j + 1)])

    # 側面（下端）
    for j in range(cols - 1):
        faces.append(
            [top(rows - 1, j), top(rows - 1, j + 1), bottom(rows - 1, j + 1)]
        )
        faces.append(
            [top(rows - 1, j), bottom(rows - 1, j + 1), bottom(rows - 1, j)]
        )

    # 側面（左端）
    for i in range(rows - 1):
        faces.append([top(i, 0), bottom(i, 0), bottom(i + 1, 0)])
        faces.append([top(i, 0), bottom(i + 1, 0), top(i + 1, 0)])

    # 側面（右端）
    for i in range(rows - 1):
        faces.append(
            [top(i, cols - 1), top(i + 1, cols - 1), bottom(i + 1, cols - 1)]
        )
        faces.append(
            [top(i, cols - 1), bottom(i + 1, cols - 1), bottom(i, cols - 1)]
        )

    # 底面
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0, v1, v2, v3 = (
                bottom(i, j),
                bottom(i, j + 1),
                bottom(i + 1, j),
                bottom(i + 1, j + 1),
            )
            faces.append([v0, v3, v1])
            faces.append([v0, v2, v3])

    faces = np.array(faces, dtype=np.int64)

    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]

    return terrain_mesh


# --- Routes -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload_nc", methods=["POST"])
def upload_nc():
    """
    nc ファイルを受け取り、間引き済み lon/lat/Z を JSON で返す。
    """
    if "ncfile" not in request.files:
        return jsonify({"error": "ncfile フィールドがありません"}), 400

    file = request.files["ncfile"]

    try:
        lons, lats, Z = process_nc(file)
    except Exception as e:
        # エラー内容をログにも返答にも出す（開発用）
        return jsonify({"error": f"nc ファイル処理中にエラー: {e}"}), 500

    return jsonify({"lons": lons, "lats": lats, "Z": Z})


@app.route("/api/export_stl", methods=["POST"])
def export_stl():
    """
    フロントエンドから lon/lat/Z と z_scale を受け取り、
    土台付き STL を生成してダウンロードさせる。
    """
    data = request.get_json(silent=True) or {}
    lons = data.get("lons")
    lats = data.get("lats")
    Z = data.get("Z")
    z_scale = float(data.get("z_scale", 1.0))

    if lons is None or lats is None or Z is None:
        return jsonify({"error": "lons/lats/Z が不足しています"}), 400

    try:
        terrain_mesh = build_solid_stl(lons, lats, Z, z_scale)
    except Exception as e:
        return jsonify({"error": f"STL 生成中にエラー: {e}"}), 500

    # 一時ファイルとして STL を保存し、送信後に削除
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp_name = tmp.name
    tmp.close()

    terrain_mesh.save(tmp_name)

    @after_this_request
    def remove_tmp(response):
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        return response

    return send_file(
        tmp_name,
        mimetype="model/stl",
        as_attachment=True,
        download_name="terrain.stl",
    )


if __name__ == "__main__":
    # Render が PORT 環境変数を設定するので、それを使う
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
