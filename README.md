# aps360-project
Student Number:
- Sophie: 1010099723   sophiezy.liu@mail.utoronto.ca
- Lucy: 1010161496     lucy.matchette@mail.utoronto.ca
- George: 1009951358   george.xue@mail.utoronto.ca

- Ran: 1009975706       ran.qi@mail.utoronto.ca
  
# Usage
- Preprocessing
     - `data_preprocessing/preprocess_readings.py`: Generate **extract and preprocess training data**
     - `data_preprocessing/preprocess_trajerror.py`: Generate **trajectory error at each timestep**



To **train

To **compare baseline with ideal trajectory/ground truth**, see `data_analysis/compare_traj.py` (Euclidean distance) and `data_analysis/compare_traj2.py` (Dynamic Time Warping)

# File Structure
- `data`: all original rosbag files, their respective unprocessed csv subtopic files, trajectory samples, training/validation/testing data
- `data_analysis`: trajectory comparison in analysis for baseline model
- `data_preprocessing`: preprocessing data for training + EKF code
- `training`: model definition, train function
- `main.py`: run training