import datetime
import itertools

import numpy as np
import pandas as pd

__author__ = 'ielemin'


class Signals:
    def __init__(self,iXs,iYs, size=1000):
        d0 = datetime.datetime(2012,2,25)
        dt = datetime.timedelta(minutes=5)
        d1 = datetime.timedelta(minutes=1)
        index=[d0+n*dt+np.random.randint(-1,1)*d1 for n in range(size)]
        
        # Actually compute them from prices
        rX = {iX:np.random.rand(size) for iX in iXs}
        rY = {iY:np.random.rand(size) for iY in iYs}
        # This is general from now on
        rs = {'X':pd.DataFrame(rX,index=index), 'Y':pd.DataFrame(rY,index=index)}
        self.data = pd.DataFrame({(cat,x):rs[cat][x] for cat in rs for x in rs[cat]})
        
    def _computeEwma(self,covDays):
        self.allcovs = pd.DataFrame({('X'+cat,xi,xj):self.data['X'][xi]*self.data[cat][xj] for (cat,__z) in self.data for (xi,xj) in itertools.product(self.data['X'],self.data[cat])})
        allewmacovs = pd.ewma(self.allcovs, halflife=covDays,min_periods=(0.1*covDays))
        self.ewmacovs = group.group.to_dailyDf(allewmacovs)
        
    def _computeBetaSR(self):
        betaSR = pd.DataFrame({('betaSR',x,y):self.ewmacovs['XY'][x][y]/self.ewmacovs['XX'][x][x] for (x,y) in self.ewmacovs['XY']})
        self.ewmacovs = pd.concat([self.ewmacovs, betaSR],axis=1)
        
    def _computeBetaMR(self):
        shapeXX = (3,3)
        shapeXY = (3,2)
        # NOTE automatically compute array shape !!!
        betaMR = pd.DataFrame((np.dot(np.linalg.inv(np.reshape(row['XX'],shapeXX)),np.reshape(row['XY'],shapeXY)).flatten() for (index,row) in self.ewmacovs.iterrows()),columns=[('betaMR',x,y) for (x,y) in self.ewmacovs['XY'].columns],index=self.ewmacovs.index)
        self.ewmacovs = pd.concat([self.ewmacovs, betaMR],axis=1)
        
    def _computeSignals(self):
        pd.DataFrame((row*self.ewmacovs['betaSR'].asof(index) for (index,row) in self.allcovs['XY'].iterrows()),columns = self.allcovs['XY'].columns, index = self.allcovs.index)
    