# Circuit Analytics

Circuit Analytics is a web-based application designed to help engineers and students analyze analog circuit datasets. By leveraging machine learning, the platform automates the training of regression models to predict circuit parameters and visualize performance metrics.

## Features

* **Dataset Upload**: Easily upload CSV files containing analog circuit performance data.
* **Automated Model Training**: Automatically trains and evaluates three different regression models:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor


* **Best Model Selection**: Automatically selects and saves the model with the highest $R^2$ score.
* **Performance Visualization**: Generates a scatter plot comparing actual vs. predicted values to assess model accuracy.
* **Interactive Predictions**: Use the trained model to input custom features and predict target circuit parameters within valid dataset ranges.
* **Prediction History**: Keeps a record of past predictions for easy reference.

## Technical Stack

* **Backend**: Flask (Python)
* **Data Processing**: Pandas, Scikit-Learn
* **Visualization**: Matplotlib
* **Model Persistence**: Joblib

## Installation

1. **Clone the repository**:
```bash
git clone <your-repository-url>
cd Minor-Project

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


*(Ensure you have `flask`, `pandas`, `scikit-learn`, `matplotlib`, and `joblib` installed).*
3. **Run the application**:
```bash
python app.py

```


4. **Access the App**: Open your web browser and navigate to `http://127.0.0.1:5000`.

## How to Use

1. **Upload**: Navigate to the homepage and upload your circuit dataset in CSV format.
2. **Select Target**: Choose the column you wish to predict.
3. **Train**: Click the train button to build the models. The application will automatically calculate the best-performing algorithm and generate an "Actual vs. Predicted" performance graph.
4. **Predict**: Once trained, use the prediction interface to input values and get accurate estimates based on your model.

---

*Developed for efficient circuit parameter analysis and modeling.*
