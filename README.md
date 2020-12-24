# Handshape predicts transitivity in silent gesture production
Predict transitivity class of gesture based on its visual characteristics

# Files
- `handshape-data.csv` : File containing (a) participant names, (b) event names, their semantic class and transitivity, and (c) handshape codes
- `convert_handshape_to_features.py` : converts Eccarius & Brentari's (2008) handshape codes (e.g., B^T-) into component features
  - finger complexity, joint complexity, flexion of the selected fingers, flexion of the unselected fingers, selected fingers, aperture change, thumb opposed, and thumb flexion
- `analysis.py` : Main classifier analysis. Uses handshape features to predict transitivity class of each gesture
