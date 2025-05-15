# IP-SLAM Pipeline

A simple two-stage Python pipeline for radar-based SLAM using AKAZE feature matching.  
It consists of:

1. **Reconstruction stage** (`reconstruct_pipeline.py`)  
   - Watches a folder of raw radar‐dump `.txt` files  
   - Parses each scan, filters out incomplete frames (in order of scan id)  
   - Saves valid frames as polar, cartesian & heatmap JPEGs  
   - If trying with test 1-3 datasets, you can utilize frame_split_sim.py and copy the resulting individual frame .txt files into /data/raw_data

2. **SLAM stage** (`slam_pipeline.py`)  
   - Watches the valid cartesian‐scan folder  
   - Every time new frames appear it runs steps 1–9 of the origianal AKAZE SLAM:  
     1. Center marker  
     2. Preprocessing  
     3. Feature matching & transformation estimation  
     4. Scan re‑orientation & GIFs  
     5. Full‐map stitching  
     6. Pixel‑thresholded stitching  
     7. Boat‑path plotting  
     8. Revert to polar, lidarize & test‐stitch2  
     9. Pose‐file combination  
   - Each run is saved under its own timestamped subfolder (`data/slam_work/run_YYYYMMDDThhmmssZ/`)