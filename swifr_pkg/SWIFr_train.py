from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from scipy.stats import norm
import math,sys,argparse, pickle, os
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn import mixture
#from matplotlib.mlab import bivariate_normal # looks like this was dropped after matplotlib 3.1
#from matplotlib.mlab import normpdf # also dropped
import matplotlib.cm as cm
import warnings
import matplotlib.cbook
#sdm imports
from scipy.stats import multivariate_normal

class AODE_train(object):
    '''
    Given directory with component_stats.txt, scenarios.txt, and subdirectory simulations/
    containing subdirectories for each classification scenario with training examples, choose
    number of Gaussian mixture components for all distributions based on BIC minima, then
    save all necessary parameters in AODE_params/ and illustrations of learned distributions
    in component_statistic_distributions/
    '''

    def __init__(self,args):
        self.retrain = args.retrain



        self.path2allstats = args.path2files
        if self.path2allstats != '' and self.path2allstats[-1] != '/':
            self.path2allstats += '/'
        self.path2allstats += 'simulations/'
        self.path2files = args.path2files
        if self.path2files != '' and self.path2files[-1] != '/':
            self.path2files += '/'
        self.path2AODE = self.path2files+'AODE_params/'


        if os.path.isdir(self.path2AODE) == False:
            os.mkdir(self.path2AODE)

        file = open(self.path2files+'classes.txt','r')
        f = file.read()
        file.close()
        self.scenarios = [x.strip() for x in f.strip().splitlines()] # neutral and sweep for 2d example


        file = open(self.path2files+'component_stats.txt','r')
        f = file.read()
        file.close()
        f = f.strip().splitlines()
        self.statlist = [x.strip() for x in f]

        self.minscores = [[] for i in range(len(self.statlist))]
        self.maxscores = [[] for i in range(len(self.statlist))]

        self.num2stat = {i:self.statlist[i] for i in range(len(self.statlist))}
        self.stat2num = {y:x for x,y in list(self.num2stat.items())}        

        self.colors = ['blue','red','green','purple','orange'] #would need to add more colors for 6+ scenarios
        self.colorspectra = [cm.Blues,cm.Reds,cm.Greens,cm.Purples,cm.Oranges] #would need to add more colors for 6+ scenarios

        self.component_nums_1D = [['n/a' for s in self.scenarios] for x in self.statlist]
        self.component_nums_2D = [[['n/a' for s in self.scenarios] for y in self.statlist] for x in self.statlist]

        if os.path.isdir(self.path2files+'BIC_plots/') == False:
            os.mkdir(self.path2files+'BIC_plots/')
        if os.path.isdir(self.path2files+'component_statistic_distributions/') == False:
            os.mkdir(self.path2files+'component_statistic_distributions')
            os.mkdir(self.path2files+'component_statistic_distributions/marginals')
            os.mkdir(self.path2files+'component_statistic_distributions/joints')
        self.read_in_all()


    def tuples(self,stat1,stat2,scenario,round1=False):
        if round1 == False:
            scores = pickle.load(open(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_tuples.p','rb'))

        else:
            print('learning '+scenario+' joint distributions for '+stat1+' and '+stat2+'...')
            scores = []

            for filename in os.listdir(self.path2allstats+scenario+'/'):
                if filename[0] != '.':
                    file = open(self.path2allstats+scenario+'/'+filename,'r')
                    f = file.read()
                    file.close()
                    f = f.strip().splitlines()
                    header = f[0].strip().split('\t')
                    f = f[1:]
                    for line in f:
                        line = line.strip().split('\t')
                        score1 = float(line[header.index(stat1)])
                        score2 = float(line[header.index(stat2)])
                        if score1 != -998 and score2 != -998:
                            scores.append((score1,score2))

            pickle.dump(scores,open(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_tuples.p','wb'), protocol=2)
        return np.array(scores)

    def singles(self,stat,scenario,round1=False):
        '''
        Method to import the simulations data that includes the SNP, distance, and various statistics for both neutral
        and sweep classes.
        '''
        if round1 == False:
            scores = pickle.load(open(self.path2AODE+stat+'_'+scenario+'_singles.p','rb'))
        else:
            print('learning '+scenario+' marginal distributions for '+stat+'...')
            scores = []
            for filename in os.listdir(self.path2allstats+scenario+'/'):
                if filename[0] != '.':
                    file = open(self.path2allstats+scenario+'/'+filename,'r')
                    f = file.read()
                    file.close()
                    f = f.strip().splitlines()
                    header = f[0].strip().split('\t')
                    f = f[1:]
                    for line in f:
                        line = line.strip().split('\t')
                        score = float(line[header.index(stat)])
                        if score != -998:
                            scores.append([score])
                        
            pickle.dump(scores,open(self.path2AODE+stat+'_'+scenario+'_singles.p','wb'), protocol=2)
        # SCORES contains the stat scores for Fst, then XP-EHH, then iHS, then DDAF
        SCORES = np.array(scores)
        # the min and max from SCORES
        minscore = min(SCORES)[0]
        maxscore = max(SCORES)[0]
        #RANGE = maxscore-minscore
        # assign the min and max from SCORES to the self.minscores and self.maxscores lists of lists
        self.minscores[self.stat2num[stat]].append(minscore)
        self.maxscores[self.stat2num[stat]].append(maxscore)
        return np.array(scores)

    def plot_bic(self,stat1,stat2,scenario):
        S = self.tuples(stat1,stat2,scenario)
        BICs_full = []
        for n in range(1,11):
            H = mixture.GaussianMixture(n_components=n,covariance_type='full')
            H.fit(S)
            #AICs_full.append(H.aic(S))
            BICs_full.append(H.bic(S))
        minbic = min(BICs_full)
        argminbic = BICs_full.index(minbic)+1
        print('number of components for '+scenario+': '+str(argminbic))
        plt.plot(list(range(1,11)),BICs_full,'o-',color='darkblue',ms=5,markeredgecolor='none')
        plt.plot(argminbic,minbic,'o-',color='coral',ms=5,markeredgecolor='red')
        plt.xlabel('number of Gaussian mixture components')
        plt.ylabel('BIC')
        plt.savefig(self.path2files+'BIC_plots/'+stat1+'_'+stat2+'_'+scenario+'_BIC.pdf')
        plt.clf()

        self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][self.scenarios.index(scenario)] = argminbic

    def plot_bic_1D(self,stat,scenario):
        S = self.singles(stat,scenario)
        BICs = []
        # the bic is being calculated for 1 to 11 components
        for n in range(1,11):
            # call sklearn gaussian mixture model
            H = mixture.GaussianMixture(n_components=n)
            # fit the mixture model based on the column(s) of data provided
            H.fit(S)
            #AICs.append(H.aic(S))
            # return the bic scores from the sklearn gaussian mixture model
            BICs.append(H.bic(S))

        minbic = min(BICs)
        argminbic = BICs.index(minbic)+1
        print('number of components for '+scenario+': '+str(argminbic))            
        plt.xlabel('number of Gaussian mixture components')
        plt.ylabel('BIC')
        plt.plot(list(range(1,11)),BICs,'o-',color='darkblue',ms=5,markeredgecolor='none')
        plt.plot(argminbic,minbic,'o-',color='coral',ms=5,markeredgecolor='red')
        plt.savefig(self.path2files+'BIC_plots/'+stat+'_'+scenario+'_BIC.pdf')
        plt.clf()
        self.component_nums_1D[self.stat2num[stat]][self.scenarios.index(scenario)] = argminbic

    def read_in_all(self):
        for stat in self.statlist:
            for scenario in self.scenarios:
                self.singles(stat,scenario,round1=True)

        for i in range(len(self.statlist)-1):
            for j in range(i+1,len(self.statlist)):
                for scenario in self.scenarios:
                    self.tuples(self.statlist[i],self.statlist[j],scenario,round1=True)

    def run_bic(self):
        for stat in self.statlist:
            print('learning number of Gaussian mixture components for '+stat)
            for scenario in self.scenarios:
                self.plot_bic_1D(stat,scenario)

        for i in range(len(self.statlist)-1):
            for j in range(i+1,len(self.statlist)):
                print('learning number of Gaussian mixture components for joint '+self.statlist[i]+', '+self.statlist[j])
                for scenario in self.scenarios:
                    self.plot_bic(self.statlist[i],self.statlist[j],scenario)

        #write marginal component_nums file
        out = open(self.path2AODE+'marginal_component_nums','w')
        header = 'statistic\t'
        for scenario in self.scenarios:
            header += scenario+'\t'
        header = header.strip()
        out.write(header+'\n')
        for i in range(len(self.component_nums_1D)):
            line = self.num2stat[i]+'\t'
            for scenario in self.scenarios:
                line += str(self.component_nums_1D[i][self.scenarios.index(scenario)])+'\t'
            line = line.strip()
            out.write(line+'\n')
        out.close()

        #write joint component_nums file
        out = open(self.path2AODE+'joint_component_nums','w')
        header = 'stat1\tstat1\t'
        for scenario in self.scenarios:
            header += scenario+'\t'
        header = header.strip()
        out.write(header+'\n')
        for i in range(len(self.component_nums_2D)-1):
            for j in range(i+1,len(self.component_nums_2D[i])):
                line = self.num2stat[i]+'\t'+self.num2stat[j]+'\t'
                for scenario in self.scenarios:
                    line += str(self.component_nums_2D[i][j][self.scenarios.index(scenario)])+'\t'
                line = line.strip()
                out.write(line+'\n')
        out.close()

    def gmm_fit(self,stat1,stat2,scenario):
        S = self.tuples(stat1,stat2,scenario)
        G = mixture.GaussianMixture(n_components=self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][self.scenarios.index(scenario)],covariance_type='full')
        G.fit(S)
        pickle.dump(G,open(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_GMMparams.p','wb'), protocol=2)
        return G

    def gmm_fit_1D(self,stat,scenario):
        S = self.singles(stat,scenario)
        G = mixture.GaussianMixture(n_components=self.component_nums_1D[self.stat2num[stat]][self.scenarios.index(scenario)])
        G.fit(S)
        pickle.dump(G,open(self.path2AODE+stat+'_'+scenario+'_1D_GMMparams.p','wb'), protocol=2)
        return G

    def plot_gmm_marginals(self,stat):
        fig = plt.figure()
        for scenario in self.scenarios:
            G = self.gmm_fit_1D(stat,scenario)
            mu = G.means_
            sigma = G.covariances_
            w = G.weights_
            minscore = min(self.minscores[self.stat2num[stat]])
            maxscore = max(self.maxscores[self.stat2num[stat]])
            x = np.linspace(minscore,maxscore,100)
            Z = 0
            for i in range(len(w)):
                # normpdf deprecated, original matplotlib code below [sdm]
                #Z = Z + w[i]*normpdf(x,mu[i],sigma[i])
                # orig code --> 1. / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (1. / sigma * (x - mu)) ** 2)
                Z = Z + w[i] * (1. / (np.sqrt(2 * np.pi) * sigma[i]) * np.exp(-0.5 * (1. / sigma[i] * (x - mu[i])) ** 2))

            plt.plot(x,Z[0],self.colors[self.scenarios.index(scenario)%len(self.colors)])
        plt.xlabel(stat)
        plt.ylabel('frequency')
        plt.legend(self.scenarios)
        plt.savefig(self.path2files+'component_statistic_distributions/marginals/'+stat+'_marginal.pdf')
        plt.clf()

    def plot_gmm_contour(self,stat1,stat2):

        fig = plt.figure()
        legendlocations = {x:[] for x in self.scenarios}
        for scenario in self.scenarios:
            G = self.gmm_fit(stat1,stat2,scenario)
            mu = G.means_
            sigma = G.covariances_
            legendlocations[scenario] = [mu[0][0]+.2*sigma[0][0][0],mu[0][1]+.2*sigma[0][1][1]]            
            w = G.weights_
            minscore1 = min(self.minscores[self.stat2num[stat1]])
            maxscore1 = max(self.maxscores[self.stat2num[stat1]])
            minscore2 = min(self.minscores[self.stat2num[stat2]])
            maxscore2 = max(self.maxscores[self.stat2num[stat2]])                            
            x = np.linspace(minscore1,maxscore1,100)
            y = np.linspace(minscore2,maxscore2,100)
            X,Y = np.meshgrid(x,y)
            Z = 0
            for i in range(len(w)):
                # bivariate_normal has been deprecated by matplotlib
                #Z = Z + w[i]*bivariate_normal(X,Y,mux=mu[i][0],muy=mu[i][1],sigmax=math.sqrt(sigma[i][0][0]),sigmay=math.sqrt(sigma[i][1][1]),sigmaxy=sigma[i][0][1])

                # deprecated matplotlib code, will retain until confirm that scipy code works correctly
                # https://github.com/matplotlib/matplotlib/blob/81e8154dbba54ac1607b21b22984cabf7a6598fa/lib/matplotlib/mlab.py#L1866
                mux = mu[i][0]
                muy = mu[i][1]
                sigmax = math.sqrt(sigma[i][0][0])
                sigmay = math.sqrt(sigma[i][1][1])
                sigmaxy = sigma[i][0][1]

                Xmu = X - mux
                Ymu = Y - muy

                rho = sigmaxy / (sigmax * sigmay)
                z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
                denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
                Z = Z + w[i] * np.exp(-z / (2 * (1 - rho ** 2))) / denom

                # multivariate normal format(mean=[vector of means], cov=[[x,xy], [xy,y]])
                # stackoverflow==> https://stackoverflow.com/questions/63623170/an-equivalent-function-to-matplotlib-mlab-bivariate-normal
                #rv = multivariate_normal(mean=[mu[i][0], mu[i][1]], cov=[[math.sqrt(sigma[i][0][0]), sigma[i][0][1]], [sigma[i][0][1], math.sqrt(sigma[i][1][1])]])
                #Z = Z + rv.pdf(np.stack((X, Y))) * w[i]

            C = plt.contour(X, Y, Z, 10, cmap=self.colorspectra[self.scenarios.index(scenario) % len(self.colorspectra)])

        plt.xlabel(stat1)
        plt.ylabel(stat2)
        for scenario in self.scenarios:
            plt.text(legendlocations[scenario][0],legendlocations[scenario][1],scenario,color=self.colors[self.scenarios.index(scenario)%len(self.colors)])
        plt.savefig(self.path2files+'component_statistic_distributions/joints/'+stat1+'_'+stat2+'_joint.pdf')
        plt.clf()

    def plot_contours(self):
        for i in range(len(self.statlist)-1):
            for j in range(i+1,len(self.statlist)):
                stat1 = self.statlist[i]
                stat2 = self.statlist[j]
                self.plot_gmm_contour(stat1,stat2)
        for stat in self.statlist:
            self.plot_gmm_marginals(stat)

    def retrain_classifier(self):
        file = open(self.path2AODE+'marginal_component_nums','r')
        f = file.read()
        file.close()
        f = f.strip().splitlines()[1:]
        for i in range(len(f)):
            f[i] = f[i].strip().split('\t')
            stat = f[i][0]
            for j in range(len(self.scenarios)):
                self.component_nums_1D[self.stat2num[stat]][j] = int(f[i][j+1])

        file = open(self.path2AODE+'joint_component_nums','r')
        f = file.read()
        file.close()
        f = f.strip().splitlines()[1:]
        for i in range(len(f)):
            f[i] = f[i].strip().split('\t')
            stat1 = f[i][0]
            stat2 = f[i][1]
            for j in range(len(self.scenarios)):
                self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][j] = int(f[i][j+2])

        self.plot_contours()

if __name__ == '__main__': # added for debugging reasons.
#def main(): # removed for debugging reasons
    
    #suppress matplotlib deprecation warnings
    #warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    # argparse is a parser for command line options
    parser = argparse.ArgumentParser()
    # parse the arguments provided by the user
    #parser.add_argument('--path',action='store',dest='path2files',default='') #path to all input files (simulations in a 'simulations' directory, and compstats, scenarios files)
    parser.add_argument('--path',action='store',dest='path2files',default='test_data/simulations_4_swifr/') # added for debugging reasons, the line commented above is the original
    parser.add_argument('--retrain',action='store_true',dest='retrain')
    # put the arguments
    args = parser.parse_args()
    #args.path2files = 'test_data/example_2classes/' # added for debugging only

    A = AODE_train(args)

    if args.retrain:
        A.retrain_classifier()
        print('Training complete. Run the command swifr_test with --path2trained '+args.path2files)
    else:
        A.run_bic()
        A.plot_contours()
        print('Training complete. Run the command swifr_test with --path2trained '+args.path2files)
