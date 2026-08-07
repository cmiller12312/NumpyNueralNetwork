import matplotlib.pyplot as plt
from dataset import Dataset

#NEEDS UPDATED TO MATCH FINISHED NN
class Visualizer:
    def drawData2D(self):
        X, y = Dataset.loadCircles()

        try:
            plt.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                c=y
            )
        except:
           plt.scatter(
                X[:, 0],
                X[:, 1],
                c=y
            ) 
        plt.show()
    
    def drawData3D(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, y = Dataset.loadAdultCensus()

        try:
            ax.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                X.iloc[:, 2],
                c=y
            )
        except:
           ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=y
            ) 
        plt.show()


if __name__ == "__main__":
    Visualizer().drawData2D()
    Visualizer().drawData3D()