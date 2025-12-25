# GADM Boundary Exporter (Streamlit)

A Streamlit dashboard that downloads GADM 4.1 administrative boundaries for any ISO3 country code, lets you search a city/region name across admin levels (0–4), previews the boundary on an OpenStreetMap map, and exports the selected boundary as GeoJSON, KML, or a zipped Shapefile.

## Features
- Auto-downloads GADM 4.1 shapefiles by ISO3 country code (cached locally per country)
- Searches across all `NAME_*` fields across admin levels 0–4
- Filter results by admin level, select one match or combine all matches
- Map preview with a clearly styled boundary polygon (OpenStreetMap only)
- Export formats:
  - GeoJSON (`.geojson`)
  - KML (`.kml`)
  - Shapefile as ZIP (`.zip`) so downloads are complete
- Clear Cache button to delete the local country cache and refresh downloads

## Project Structure
