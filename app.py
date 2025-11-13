from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import xarray as xr
import io
from stl import mesh

app = Flask(__name__)

# --- Utility functions ---
def process_nc(file_stream):
    ds = xr.open_dataset(file_stream)
    cand_vars = ['elevation', 'z', 'bathymetry', 'bedrock']
    varname = None
    for v in cand_vars:
        if v in ds.data_vars:
            varname = v
            break
    if varname is None:
        if len(ds.data_vars) == 1:
            varname = list(ds.data_vars)[0]
        else:
            raise ValueError("標高データが見つかりません。")

    da = ds[varname]
    lon_name = [c for c in da.dims if 'lon' in c.lower()][0]
    lat_name = [c for c in da.dims if 'lat' in c.lower()][0]
    lons = ds[lon_name].values
    lats = ds[lat_name].values
    Z = da.values

    max_n = 800
    step_y = max(1, int(np.ceil(Z.shape[0] / max_n)))
    step_x = max(1, int(np.ceil(Z.shape[1] / max_n)))

    Z_sub = Z[::step_y, ::step_x]
    lats_sub = lats[::step_y]
    lons_sub = lons[::step_x]

    return lons_sub.tolist(), lats_sub.tolist(), Z_sub.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload_nc", methods=["POST"])
def upload_nc():
    if "ncfile" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["ncfile"]
    lons, lats, Z = process_nc(file)

    return jsonify({
        "lons": lons,
        "lats": lats,
        "Z": Z
    })


@app.route("/api/export_stl", methods=["POST"])
def export_stl():
    data = request.json
    lons = np.array(data["lons"])
    lats = np.array(data["lats"])
    Z = np.array(data["Z"])
    z_scale = float(data["z_scale"])

    avg_lat_rad = np.deg2rad(lats.mean())
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(avg_lat_rad)

    lon_origin = lons.min()
    lat_origin = lats.min()

    Z_scaled = Z * z_scale

    rows, cols = Z.shape
    num_top = rows * cols
    total_vertices = num_top * 2
    vertices = np.zeros((total_vertices, 3))

    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            x = (lons[j] - lon_origin) * meters_per_degree_lon
            y = (lats[i] - lat_origin) * meters_per_degree_lat
            z = Z_scaled[i, j]
            vertices[idx] = [x, y, z]

    min_z_surface = Z_scaled.min()
    base_thickness = 100
    bottom_z = min_z_surface - base_thickness

    for i in range(rows):
        for j in range(cols):
            idx_top = i*cols + j
            idx_bottom = num_top + idx_top
            vertices[idx_bottom] = [vertices[idx_top][0],
                                    vertices[idx_top][1],
                                    bottom_z]

    def top(i,j): return i*cols + j
    def bottom(i,j): return num_top + i*cols + j

    faces = []

    for i in range(rows-1):
        for j in range(cols-1):
            v0, v1, v2, v3 = top(i,j), top(i,j+1), top(i+1,j), top(i+1,j+1)
            faces.append([v0,v1,v3])
            faces.append([v0,v3,v2])

    for j in range(cols-1):
        faces.append([top(0,j), bottom(0,j), bottom(0,j+1)])
        faces.append([top(0,j), bottom(0,j+1), top(0,j+1)])

    for j in range(cols-1):
        faces.append([top(rows-1,j), top(rows-1,j+1), bottom(rows-1,j+1)])
        faces.append([top(rows-1,j), bottom(rows-1,j+1), bottom(rows-1,j)])

    for i in range(rows-1):
        faces.append([top(i,0), bottom(i,0), bottom(i+1,0)])
        faces.append([top(i,0), bottom(i+1,0), top(i+1,0)])

    for i in range(rows-1):
        faces.append([top(i,cols-1), top(i+1,cols-1), bottom(i+1,cols-1)])
        faces.append([top(i,cols-1), bottom(i+1,cols-1), bottom(i,cols-1)])

    for i in range(rows-1):
        for j in range(cols-1):
            v0, v1, v2, v3 = bottom(i,j), bottom(i,j+1), bottom(i+1,j), bottom(i+1,j+1)
            faces.append([v0,v3,v1])
            faces.append([v0,v2,v3])

    faces = np.array(faces)

    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]

    buffer = io.BytesIO()
    terrain_mesh.save(buffer)
    buffer.seek(0)

    return send_file(buffer, mimetype="application/octet-stream",
                     as_attachment=True, download_name="terrain.stl")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
