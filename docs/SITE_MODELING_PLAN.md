# Gary micro-twin — site modeling (footprints & optional 3D assets)

## Sites

| Site ID | Display name | Default geometry |
|---------|--------------|------------------|
| `city_hall` | Gary City Hall | Multi-vertex civic outline (simplified map-aligned shape) |
| `public_library` | Gary Public Library & Cultural Center | L-shaped public-building outline |
| `west_side_leadership` | West Side Leadership Academy | Campus envelope (single outer ring) |

Footprints are **approximate** for visualization. Replace with **authoritative GIS** (parcel/building layer) or **OSM** exports when available.

## Current status

- **Default:** **Extruded `PolygonLayer`** footprints from `gary_site_geometry.py` (not axis-aligned rectangles).
- **Optional:** **`ScenegraphLayer`** + local **`.glb`** when `configs/wireless_scene/site_models.json` points at an existing file under `assets/models/` (or another repo-relative path).
- **Fallback:** If a GLB path is configured but the file is **missing**, or `ScenegraphLayer` fails to build, the site **falls back** to the extruded footprint automatically.

## Config files

### `configs/wireless_scene/site_footprints.json`

```json
{
  "version": 1,
  "sites": {
    "city_hall": {
      "polygon": [[-87.338, 41.5842], [-87.3376, 41.5842], [-87.3376, 41.5838], [-87.338, 41.5838], [-87.338, 41.5842]],
      "height_m": 55,
      "footprint_source_note": "Replaced from city GIS export (example)."
    }
  }
}
```

- **`polygon`:** closed ring `[lon, lat]` per vertex (first point should equal last).
- **`height_m`**, **`risk_bias`**, optional string fields: merged over built-in defaults.

### `configs/wireless_scene/site_models.json`

```json
{
  "version": 1,
  "sites": {
    "city_hall": {
      "glb_relative": "assets/models/gary_city_hall.glb",
      "scale": 1.0,
      "orientation_deg": [0, 0, 0],
      "anchor_offset_lonlat": [0.0, 0.0]
    }
  }
}
```

- **`glb_relative`:** path relative to **repo root**.
- **`scale`:** scalar or `[sx, sy, sz]`; drives `size_scale` in `ScenegraphLayer` (heuristic).
- **`orientation_deg`:** passed to deck; behavior depends on pydeck/deck.gl version (tune locally).
- **`anchor_offset_lonlat`:** shift from computed footprint **centroid** (degrees).

### Expected asset paths (optional)

- `assets/models/gary_city_hall.glb`
- `assets/models/gary_library.glb`
- `assets/models/west_side_leadership_academy.glb`

These files are **not** committed by default. Add your own models locally.

## Runtime notes

- **Local `streamlit run`:** `ScenegraphLayer` uses `file://` URIs; some browsers restrict mixed content — if the model does not appear, rely on **extruded footprints** or serve the GLB over `http(s)`.
- **Streamlit Cloud:** file URLs may be blocked; treat **3D assets as optional**; footprints remain the portable baseline.

## Visual distinction (UI)

- **Site-specific outline colors** (civic / library / school) in addition to coexistence risk fill.
- **Labels** use `map_label` (emoji + short name) for quick recognition.

See also `docs/MICROTWIN_REALISM_PLAN.md` and `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md`.
