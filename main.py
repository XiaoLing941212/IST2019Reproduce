from sway import sway_runner
from function import generate_table, fitnessFunctionTime_detectedMutants
import time
from scipy import io
from metric import is_pareto_efficient, calcHV, calcMSTET
import csv



def loadTCDData(project):
    if project == "Twotanks":
        filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/TCData.mat"
    elif project == "ACEngine":
        filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/FDC_DATA.mat"
    elif project == "EMB":
        filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/BlackBoxMetrics_2.mat"
    elif project == "CW":
        filePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/TC_time.mat"
    
    return io.loadmat(filePath)

def main():
    # set project and read TCD data
    project = "Twotanks"

    # set param for each project
    if project == "Twotanks":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 11, 7
        number_tc = 150
        time_metric = [TCD['TCData'][0][i][0][0][0] for i in range(number_tc)]
    elif project == "ACEngine":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 4, 1
        number_tc = 120
        time_metric = [TCD['test_case'][0][i][1][0][0] for i in range(number_tc)]
    elif project == "EMB":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 1, 1
        number_tc = 150
        time_metric = [TCD['TCData'][0][i][0][0][-1][0][0] for i in range(number_tc)]
    elif project == "CW":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 15, 4
        number_tc = 133
        time_metric = [TCD['time_testCases'][i][0] for i in range(number_tc)]

    repeat = 1

    writePath = "/mnt/e/Research/IST2019Paper/IST2019PY/result/" + str(project) + "_test.csv"

    with open(writePath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['hv', 'ms_tet', 'time'])

        # repeat n times
        for _ in range(repeat):
            # generate samples
            pop_table = generate_table(time_metric, project, nInputs, nOutputs, number_tc)

            for row in pop_table:
                print(row[1])
            return

            # SWAY selects the representative data
            start_time = time.time()
            res = sway_runner(pop_table)
            finish_time = time.time()
            print("SWAY execution time: ", finish_time-start_time)

            ### calculate evaluation metric - hv and average weighted sum of mutation score and normalized test execution time
            G = []
            for item in res:
                t, m = fitnessFunctionTime_detectedMutants(item[0], time_metric, project)
                G.append([t, m])
            
            # pareto set
            P = is_pareto_efficient(G)

            # calculate hypervolume
            hv = calcHV(G, P)

            # calculate mean test execution time and mean mutation score
            ms_tet = calcMSTET(G)
            
            writer.writerow([hv, ms_tet, finish_time-start_time])



if __name__ == '__main__':
    main()