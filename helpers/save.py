import os
import re
import json

from fuzzywuzzy import fuzz
from pathlib import Path
# make sure you installed fuzzywuzzy and python-Levenshtein

class results():

    def __init__(self, filename = None):
        if not filename:
            filename = input('You did not pass in file name, please input your file name (please include .json as file extension)')
            while filename.search('\.(json)$', filename) == None:
                filename = input('Your input does not contain .json as file extension. Please input correct file name')

        self.filepath = os.path.join(Path().absolute().parents[1], 'results', filename)
        with open(self.filepath, 'r+') as results:
            try:
                self.results = json.load(results)
            except json.decoder.JSONDecodeError:
                self.results = {}
                pass

    def save(self, test = None, results = None, additional = None, repeat = False):
        if test == None:
            return ValueError('Test name cannot be empty')
        if results == None:
            return ValueError('Results cannot be empty. If no results, please pass in \{\} instead')

        allLower = [x.lower() for x in self.results]

        if test.lower() not in allLower:
            self.results[test] = {}
            self.results[test]['results'] = results
            if additional:
                self.results[test] = {**self.results[test], **additional}
        else:
            if repeat:
                self.results[test]['results'] = results
                if additional:
                    self.results[test] = {**self.results[test], **additional}
                # print(f'Test {test} is already in record, proceed to override.')
            else:
                ans = input(f'Test {test} is already in record, would you like to override? (y/n)')
                while ans.lower() not in ['y', 'n', 'yes', 'no']:
                    ans = input(f'answer {ans} invalid, please choose (y/n)')
                if ans.lower() in ['y', 'yes']:
                    self.results[test]['results'] = results
                    if additional:
                        self.results[test] = {**self.results[test], **additional}
                else:
                    print('Do not override, test result not saved')
                    return

        with open(self.filepath, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)
            print('record saved')

    def checkIfTestExist(self, search = None):
        if search == None:
            return ValueError('Test name cannot be empty')

        curTest = []
        for test in self.results:
            if fuzz.token_sort_ratio(test, search) >= 50:
                curTest.append((test, fuzz.token_sort_ratio(test, search)))
        
        curTest = sorted(curTest, key=lambda x: x[1], reverse=True)
        if len(curTest) == 0:
            print('Cannot find any similar test')
            return
        else:
            print('Found the following similar tests: ...')
            for t in curTest:
                print(t[0])

    
    def refresh(self):
        with open(self.filepath) as results:
            self.results = json.load(results)

        return self.results

    def printExistingTests(self):
        if len(self.results.keys()) == 0:
            print('There are currently no tests')
        else:
            for test in self.results:
                print(test)

    def getExistingTests(self):
        if len(self.results.keys()) == 0:
            print('There are currently no tests')
        else:
            return [test for test in self.results]

    def getAllResults(self):
        if len(self.results.keys()) == 0:
            print('There are currently no tests')
        return self.results


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