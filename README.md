 KNN Handwritten Digit Classification
An interactive web-based implementation of the K-Nearest Neighbors (KNN) algorithm for handwritten digit classification, built entirely with vanilla HTML, CSS, and JavaScript.
📋 Project Overview
This project demonstrates the KNN machine learning algorithm on a simulated handwritten digits dataset (similar to sklearn's load_digits()). The application provides a comprehensive visualization of the classification process, including accuracy metrics, confusion matrices, and real-time predictions.
✨ Features

Interactive Dataset Visualization: View sample digit images from the training set
Flexible K Selection: Train models with different K values (3, 5, 7, 9, 11, 13, 15)
Real-time Training: Train individual K values or batch train all at once
Performance Metrics: Track accuracy, best K value, and training time
Accuracy vs K Graph: Visual representation of how accuracy changes with K
Confusion Matrix: Detailed 10x10 matrix showing classification results
Prediction Display: View 15 random test samples with predictions
Feature Scaling: Implements StandardScaler for normalized features
Educational Content: Built-in interview Q&A section

🚀 How to Use

Open the Application: Open index.html in any modern web browser
View Dataset: Sample digit images are automatically displayed
Select K Value: Choose a K value from the dropdown (default: 3)
Train Model: Click "Train Model" to train with selected K
Compare K Values: Click "Train All K Values" to test multiple K values
Analyze Results: View accuracy chart, confusion matrix, and predictions
Reset: Click "Reset" to clear all results and start over

🛠️ Technical Implementation
Algorithm Components
Data Generation

Simulated 1,800 samples (180 per digit class)
8x8 pixel images (64 features)
Base patterns with random noise for variation
80/20 train-test split

Feature Scaling (StandardScaler)
javascriptscaled_value = (value - mean) / standard_deviation

Ensures all features contribute equally
Critical for distance-based algorithms

Euclidean Distance
javascriptdistance = √[(a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²]
KNN Classification

Calculate distances to all training samples
Sort distances and select K nearest neighbors
Vote: assign most common class among K neighbors

File Structure
knn-digit-classification/
│
├── index.html          # Main HTML structure
├── styles.css          # Complete styling and responsive design
├── script.js           # KNN algorithm implementation and visualization
└── README.md           # Project documentation
📊 Key Concepts Demonstrated
1. What is K in KNN?
K is the number of nearest neighbors considered when making predictions. The algorithm finds K training samples closest to the test point and assigns the most common class among those K neighbors.
2. Why Scaling is Required for KNN?
KNN uses distance metrics (like Euclidean distance) to find nearest neighbors. Features with larger scales can dominate the distance calculation, making the algorithm biased. StandardScaler ensures all features contribute equally by transforming them to have mean=0 and variance=1.
3. What is Euclidean Distance?
Euclidean distance is the straight-line distance between two points in n-dimensional space. It's the most common distance metric used in KNN for continuous features.
4. What Happens if K is Too Low?

Too Low (e.g., K=1): Model becomes sensitive to noise and outliers (overfitting)
Too High: Decision boundary becomes oversimplified (underfitting)
Optimal K: Found through cross-validation and testing multiple values

5. Limitations of KNN

Computational Cost: Slow prediction time (must compute distances to all training samples)
Memory Intensive: Stores entire training dataset (lazy learning)
Curse of Dimensionality: Performance degrades with high-dimensional data
Sensitive to Irrelevant Features: Requires feature selection/engineering
Imbalanced Data: Biased toward majority class

🎯 Learning Outcomes
After using this application, you will understand:

Distance-based classification algorithms
Importance of feature scaling in machine learning
Hyperparameter tuning (K value optimization)
Model evaluation using confusion matrices
Trade-offs between different K values
Real-world application of KNN algorithm

📈 Performance Metrics
The application tracks and displays:

Accuracy: Percentage of correct predictions
Confusion Matrix: Detailed breakdown of predictions vs actual labels
Accuracy vs K Chart: Visualization of model performance across K values
Training Time: Algorithm execution time
Best K Value: Optimal K based on accuracy

🎨 User Interface Features

Responsive Design: Works on desktop, tablet, and mobile devices
Interactive Charts: Real-time visualization of results
Color-Coded Matrix: Easy-to-read confusion matrix with gradient colors
Sample Visualizations: View actual digit images and predictions
Smooth Animations: Professional transitions and hover effects

🔧 Browser Compatibility

Chrome (recommended)
Firefox
Safari
Edge
Opera

📝 Interview Preparation
The application includes a comprehensive FAQ section covering common interview questions about KNN:

Algorithm fundamentals
Feature scaling rationale
Distance metrics
Hyperparameter selection
Algorithm limitations

🤝 Contributing
This is an educational project. Feel free to fork, modify, and enhance it for your learning purposes.
📄 License
Open source educational project. Free to use and modify.
🎓 Educational Use
Perfect for:

Machine learning beginners
Computer science students
Interview preparation
Teaching demonstrations
Portfolio projects

🌟 Key Highlights
✅ No Dependencies: Pure vanilla JavaScript implementation
✅ Visual Learning: Interactive charts and visualizations
✅ Complete Implementation: Full KNN algorithm from scratch
✅ Educational: Built-in learning resources and explanations
✅ Production-Ready Code: Clean, well-documented, and maintainable
📞 Support
For questions or issues, please refer to the interview Q&A section in the application or consult machine learning documentation.
<img width="1266" height="960" alt="Screenshot 2026-01-30 142747" src="https://github.com/user-attachments/assets/2c41a6d8-2029-4889-b86d-0027ebf91599" />
<img width="1068" height="912" alt="Screenshot 2026-01-30 142834" src="https://github.com/user-attachments/assets/84808c1a-3f84-48be-9bea-8495080471a1" />


