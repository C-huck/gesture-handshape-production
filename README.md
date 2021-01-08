# Handshape predicts transitivity in silent gesture production
Precis:
- This project investigates whether silent gestures can be shown to contain subunit structure, by way of argument marking morphology.
- 432 silent gestures were elicited from 6 participants representing 72 action videos (36 involve the movement of an agent/object and 36 involve the manipulation of an object).  
- Gestures were annotated for handshape
- Handshape features were used as predictors to classify whether gesture is intransitive or transitive
- 

See more detailed abstract [here](https://github.com/C-huck/C-huck.github.io/blob/master/pdfs/CUNY2020.pdf)
See manuscript [here](https://docs.google.com/document/d/1OiYknCBRLQxD7zS68Q_yIqU_QCEYDRbUg4jz_H-aHYg/edit?usp=sharing)

# Files
- `handshape-data.csv` : Dataset containing (a) participant names; (b) event names, their semantic class and transitivity; and (c) handshape codes. Handshape codes from Eccarius & Brentari, 2008, Sign Language & Linguistics.
- `convert_handshape_to_features.py` : Converts Eccarius & Brentari's (2008) handshape codes (e.g., B^T-) into component features
  - finger complexity, joint complexity, flexion of the selected fingers, flexion of the unselected fingers, selected fingers, aperture change, thumb opposed, and thumb flexion
- `analysis.py` : Main classifier analysis. Uses handshape features to predict transitivity class of each gesture
