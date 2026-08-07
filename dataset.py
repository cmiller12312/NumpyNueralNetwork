import sklearn.datasets as skl
import pandas as pd

class Dataset:
    @staticmethod
    def loadCircles(samples=200, shuffle=True, noise=0.05, randomState=42):
        return skl.make_circles(n_samples=samples, shuffle=shuffle, noise=noise, random_state=randomState)
    
    @staticmethod
    def loadIris():
        return skl.load_iris(return_X_y=True)
    
    @staticmethod
    def loadBreastCancer():
        return skl.load_breast_cancer(return_X_y=True)
    
    @staticmethod
    def loadWine():
        return skl.load_wine(return_X_y=True)
    
    @staticmethod
    def loadDiabetes():
        return skl.load_diabetes(return_X_y=True)
    
    @staticmethod
    def loadLinnerrud():
        return skl.load_linnerud(return_X_y=True)
    
    @staticmethod
    def loadDigits():
        return skl.load_digits(return_X_y=True)
    
    @staticmethod
    def loadAdultCensus():
        df = pd.read_csv("datasets/adult.csv")

        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        y = df["income"].map({
            "<=50K": 0,
            ">50K": 1
        })

        X = df.drop("income", axis=1)

        X = pd.get_dummies(X)

        X = X.astype(float)

        return X.to_numpy(), y.to_numpy()
    
    @staticmethod
    def SimpleStrToInt(entry):
        return (int.from_bytes(entry.encode('utf-8'), byteorder='big', signed=False) % 10)

if __name__ == "__main__":
    Dataset.loadAdultCensus()