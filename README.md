# Team 4


Introduction to Human and Computer Vision project code 

- Sara Cela Alfonso
- Zuzanna Szafranowska
- Manuel Rey Area
- Grzegorz Skorupko

<h2>Usage</h2>

To run this code, execute the following:
```ssh
python3 traffic_sign_detection.py <dataset_path> <output_masks_path> <pixel_method> [--windowMethod=<wm>] [--calculateMetrics] 
```
- <dataset_path> - path to the dataset with images
- <output_masks_path> - where to save the output masks and windows
- <pixel_method> - select method for selecting pixel candidates
- [--windowMethod=<wm>] - select method for selecting window candidates (optional)
- [--calculateMetrics] - flag to turn on calculating metrics (optional)