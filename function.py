from scipy import io
import random
import pandas as pd
import numpy as np

def loadMatData(file):
    return io.loadmat(file)

def loadTCDData(project):
    if project == "Twotanks":
        filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/TCData.mat"
    
    return io.loadmat(filePath)

def allTestCases_Derivative(project, activationArray):
    filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/derivative.mat"
    derivative = loadMatData(filePath)['derivative']

    newDerivative, allDerivative = 0, 0

    for k in range(len(activationArray)):
        allDerivative += derivative[k][0]

        if activationArray[k] == 1:
            newDerivative += derivative[k][0]
    
    return newDerivative/allDerivative

def allTestCases_Infinite(project, activationArray):
    filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/infinite.mat"
    infinite = loadMatData(filePath)['infinite']

    newInf, allInf = 0, 0

    for k in range(len(activationArray)):
        allInf += infinite[k][0]

        if activationArray[k] == 1:
            newInf += infinite[k][0]
    
    return newInf/allInf

def allTestCases_Instability(project, activationArray):
    filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/instability.mat"
    instability = loadMatData(filePath)['instability']

    newInstability, allInstability = 0, 0

    for k in range(len(activationArray)):
        allInstability += instability[k][0]

        if activationArray[k] == 1:
            newInstability += instability[k][0]
    
    return newInstability/allInstability

def allTestCases_MinMax(project, activationArray):
    filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/minMax.mat"
    minmax = loadMatData(filePath)['minMax']

    newMinMax, allMinMax = 0, 0

    for k in range(len(activationArray)):
        allMinMax += minmax[k][0]

        if activationArray[k] == 1:
            newMinMax += minmax[k][0]

    return newMinMax/allMinMax

def calculate_Euclidean(activationArray, euclideanTable):
    NTC = 0
    outEuclidean = 0

    for i in range(len(activationArray)):
        if activationArray[i] != 0:
            NTC += 1
    
    if NTC > 1:
        for i in range(len(activationArray)):
            if activationArray[i] == 1:
                for j in range(len(activationArray)):
                    if activationArray[j] == 1:
                        outEuclidean += abs(euclideanTable[i][i] - euclideanTable[i][j])

        outEuclidean = outEuclidean / (len(activationArray) * ((len(activationArray) - 1) / 2))
    else:
        outEuclidean = 0
    
    return outEuclidean

def totalInputEuclidean(activationArray, inputE, nInputs):
    outInputEuclidean = []
    for i in range(nInputs):
        outInputEuclidean.append(calculate_Euclidean(activationArray, inputE[i]))
    
    a = sum(outInputEuclidean)

    return a / len(outInputEuclidean)

def totalOutputEuclidean(activationArray, outputE, nOutputs):
    outOutputEuclidean = []
    for i in range(nOutputs):
        outOutputEuclidean.append(calculate_Euclidean(activationArray, outputE[i]))
    
    a = sum(outOutputEuclidean)

    return a / len(outOutputEuclidean)

def costFunction(x, time_metric, number_tc, project, inputE, nInputs, outputE, nOutputs):
    time = 0
    totalTime = 0

    for i in range(number_tc):
        totalTime += time_metric[i]

        if x[i] == 1:
            time += time_metric[i]
    
    cost = time/totalTime
    discontinuity = allTestCases_Derivative(project, x)
    infinity = allTestCases_Infinite(project, x)
    instability = allTestCases_Instability(project, x)
    minmax = allTestCases_MinMax(project, x)
    # indistance = totalInputEuclidean(x, inputE, nInputs)
    # outdistance = totalOutputEuclidean(x, outputE, nOutputs)

    # return [cost, 1-discontinuity, 1-infinity, 1-instability, 1-minmax, 1-indistance, 1-outdistance]
    return [cost, 1-discontinuity, 1-infinity, 1-instability, 1-minmax]

def generate_table(time_metric, project, nInputs, nOutputs, number_tc):
    # collect input data
    inputE = []
    for i in range(nInputs):
        inputFilePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/inputEuclidean_" + str(i+1) + ".mat"
        inputE.append(loadMatData(inputFilePath)['inputEuclidean'])

    # collect output data
    outputE = []
    for i in range(nOutputs):
        outputFilePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/outputEuclidean_" + str(i+1) + ".mat"
        outputE.append(loadMatData(outputFilePath)['OutputEuclidean'])

    # define parameters  # need to change to 10000
    nPop = 1000

    # generate population table
    popTable = []
    for p in range(nPop):
        # if p%100 == 0:
        #     print("generate ", p)
        tempArray = [random.randint(0, 1) for _ in range(number_tc)]
        cost = costFunction(tempArray, time_metric, number_tc, project, inputE, nInputs, outputE, nOutputs)
        popTable.append((tempArray, cost))
        
    return popTable

def detectedMutants(activationArray, project):
    filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/MutantsMatrixTable.xlsx"
    mutantsTable = pd.read_excel(filePath).to_numpy()

    detectedMutantsArray = []

    k = 1
    a, b = len(mutantsTable), len(mutantsTable[0])

    for i in range(len(activationArray)):
        if activationArray[i] == 1:
            for j in range(b):
                if mutantsTable[i][j] == 1:
                    detectedMutantsArray.append(j)
    
    detectedMutantsArray = np.unique(np.array(detectedMutantsArray))
    
    return len(detectedMutantsArray)/b

def fitnessFunctionTime_detectedMutants(activationArray, time_metric, project):
    popSize = len(activationArray)

    time = 0
    totalTime = 0

    for i in range(len(time_metric)):
        totalTime += time_metric[i]

        if activationArray[i] == 1:
            time += time_metric[i]
    
    detectedMutantsP = detectedMutants(activationArray, project)

    return time/totalTime, 1 - detectedMutantsP

def main():
    t = generate_table()

if __name__ == "__main__":
    main()