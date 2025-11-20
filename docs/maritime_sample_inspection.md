## Inspecting maritime samples for anchor metadata

Folium visualization falls back to the default inland center `(30.0, 120.0)`
when the preprocessed samples do not contain latitude/longitude anchors
(`origin_lat/origin_lon` or `ref_lat/ref_lon/ref_theta`). Use the helper below
to confirm whether a sample includes these fields.

### Usage

```bash
python scripts/inspect_maritime_sample.py \
  data/processed_with_map/val/scene_POS_OK_2024-07-01_Waigaoqiao_Port_processed_batches_idx100_pid3460776_part0000_0.pt
```

The script prints the top-level keys plus the values of `origin_lat/origin_lon`
and `ref_lat/ref_lon/ref_theta` when present. If `metadata` or `scene_info`
are missing, the output highlights the absence so you know the sample lacks the
anchors required by `visualize_scenes_folium.py`.
