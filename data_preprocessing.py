import pandas as pd
import numpy as np
import csv


def dataProcessing(initialPath):

    data_rows = []

    for i in range(1998,2018):

        filePath = initialPath + str(i) + '.op'

        with open(filePath, 'r') as year_data:

            variables = year_data.readline().split()
            for line in year_data:
                line = line.replace('*', '')
                day_val = np.array(line.split(),dtype=str)
                vals = np.delete(day_val, (4, 6, 8, 10, 12, 14))
                data_rows.append(vals)


    filePath = initialPath +'1998.op'
    with open(filePath, 'r') as year_data:
        variables = year_data.readline().split()
        temperature_df = pd.DataFrame(data_rows, columns=variables)
        temperature_df.update(temperature_df[['FRSHTT']].applymap('"{}"'.format))
        temperature_df.to_csv('temperature_data.csv', index=False)


#, index=False, quotechar='"', header=None, quoting=csv.QUOTE_NONNUMERIC)




