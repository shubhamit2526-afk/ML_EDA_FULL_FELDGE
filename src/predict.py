from src.modeling.predict import predict_default_probability

if __name__ == "__main__":
    result = predict_default_probability()
    print(result[["Predicted_Default_Prob", "Predicted_Status", "Model", "Model_Version"]].head(10))
