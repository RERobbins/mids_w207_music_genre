import os
import re
import json

import pandas as pd

from fuzzywuzzy import fuzz
from pathlib import Path
# make sure you installed fuzzywuzzy and python-Levenshtein

class results():

    def __init__(self, filename = None):
        if not filename:
            filename = input('You did not pass in file name, please input your file name (please include .json as file extension)')
            while filename.search('\.(json)$', filename) == None:
                filename = input('Your input does not contain .json as file extension. Please input correct file name')

        self.filepath = os.path.join(Path(__file__).parents[1], 'results', filename)
        print(self.filepath)
        try:
            with open(self.filepath, 'x+') as results:
                print("File do not exist, creating file now")
                self.results = {
                    "experiment": []
                }
                self.df = pd.json_normalize(self.results, record_path=['experiment'], max_level=0)
        except FileExistsError:
            with open(self.filepath, 'r') as results:
                print("file exist, reading json")
                self.results = json.load(results)
                self.df = pd.json_normalize(self.results, record_path=['experiment'], max_level=0)
        

    def save(self, model = None, dataset = None, phase = None, results = None, additional = None, repeat = False):
        if model == None:
            return ValueError('Model name cannot be empty')
        if dataset == None:
            return ValueError('Dataset name cannot be empty')
        if phase == None:
            return ValueError('Phase name cannot be empty')
        if results == None:
            return ValueError('Results cannot be empty. If no results, please pass in \{\} instead')

        allLower = [f"{x['model'].lower()}_{x['dataset'].lower()}_{x['phase'].lower()}" for x in self.results['experiment']]

        if f'{model.lower()}_{dataset.lower()}_{phase.lower()}' not in allLower:
            res = {}
            res['model'] = model
            res['dataset'] = dataset
            res['phase'] = phase
            res['result'] = results
            if additional:
                res = {**res, **additional}

            temp_df = pd.DataFrame([res])
            self.df = pd.concat([self.df, temp_df])
        else:
            if repeat:
                self.df = self.df.drop(
                    self.df[
                        (self.df['model'] == model) &
                        (self.df['dataset'] == dataset) &
                        (self.df['phase'] == phase)
                    ].index
                )
                res = {}
                res['model'] = model
                res['dataset'] = dataset
                res['phase'] = phase
                res['result'] = results
                if additional:
                    res = {**res, **additional}

                temp_df = pd.DataFrame([res])
                self.df = pd.concat([self.df, temp_df])
                # print(f'Test {test} is already in record, proceed to override.')
            else:
                ans = input(f'Model {model} using dataset {dataset} and phase {phase} is already in record, would you like to override? (y/n)')
                while ans.lower() not in ['y', 'n', 'yes', 'no']:
                    ans = input(f'answer {ans} invalid, please choose (y/n)')
                if ans.lower() in ['y', 'yes']:
                    self.df = self.df.drop(
                        self.df[
                            (self.df['model'] == model) &
                            (self.df['dataset'] == dataset) &
                            (self.df['phase'] == phase)
                        ].index
                    )
                    res = {}
                    res['model'] = model
                    res['dataset'] = dataset
                    res['phase'] = phase
                    res['result'] = results
                    if additional:
                        res = {**res, **additional}

                    temp_df = pd.DataFrame([res])
                    self.df = pd.concat([self.df, temp_df])
                else:
                    print('Do not override, test result not saved')
                    return

        temp_result = self.df.to_json(orient="records")
        self.results['experiment'] = json.loads(temp_result)

        with open(self.filepath, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)
            print('record saved')

    def checkIfTestExist(self, model = None, dataset = None, phase = None):
        if model == None:
            return ValueError('Model name cannot be empty')
        if dataset == None:
            return ValueError('Dataset name cannot be empty')
        if phase == None:
            return ValueError('Phase name cannot be empty')

        allLower = [f"{x['model'].lower()}\{x['dataset'].lower()}\{x['phase'].lower()}" for x in self.results['experiment']]
        search = f'{model.lower()}\{dataset.lower()}\{phase.lower()}'
        curTest = []
        for test in allLower:
            if fuzz.token_sort_ratio(test, search) >= 50:
                curTest.append((test, fuzz.token_sort_ratio(test, search)))
        
        curTest = sorted(curTest, key=lambda x: x[1], reverse=True)
        if len(curTest) == 0:
            print('Cannot find any similar test')
            return
        else:
            print('Found the following similar tests: ...')
            for t in curTest:
                temp = t[0].split('\\')
                print('-------------------------------')
                print(f'model: {temp[0]}')
                print(f'dataset: {temp[1]}')
                print(f'phase: {temp[2]}')

    
    def refresh(self, returnDict = True):
        with open(self.filepath) as results:
            self.results = json.load(results)
            self.df = pd.json_normalize(self.results, record_path=['experiment'], max_level=0)

        if returnDict:
            return self.results
        else:
            return self.df

    def unnestDataframeLevel(self, level = 0):
        return pd.json_normalize(self.results, record_path=['experiment'], max_level=level)

    def printExistingTests(self):
        if len(self.results['experiment']) == 0:
            print('There are currently no tests')
        else:
            for test in self.results['experiment']:
                print('-------------------------------')
                print(f'model: {test["model"]}')
                print(f'dataset: {test["dataset"]}')
                print(f'phase: {test["phase"]}')

    # def getExistingTests(self):
    #     if len(self.results.keys()) == 0:
    #         print('There are currently no tests')
    #     else:
    #         return [test for test in self.results]

    def getAllResults(self):
        if len(self.results.keys()) == 0:
            print('There are currently no tests')
        return self.results['experiment']

    def getDataframe(self):
        return self.df


    # def getResultRanked(self, metricName = None):
    #     if metricName == None:
    #         return ValueError('Metric name cannot be empty')

    #     s = []
    #     noRes = []

    #     if len(self.results.keys()) == 0:
    #         print('There are currently no tests')
    #         return

    #     for test in self.results:
    #         if metricName in self.results[test]['results']:
    #             metricVal = self.results[test]['results'][metricName]
    #             s.append((test, metricVal))
    #         else:
    #             metricVal = 'Record does not exist'
    #             noRes.append((test, metricVal))

    #     s = sorted(s, key=lambda x: x[1], reverse=True)
    #     noRes = sorted(noRes, key=lambda y: y[0].lower())

    #     return s + noRes