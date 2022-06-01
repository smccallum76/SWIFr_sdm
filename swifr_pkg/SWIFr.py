from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import input
from builtins import range
from past.utils import old_div
from builtins import object
#from SWIFr_train import AODE_train
from sklearn import mixture
import os, pickle, math, sys, numpy as np, argparse

sys.path.append(os.getcwd())

class Mix1D(object):
    def __init__(self):
        self.means_ = []
        self.weights_ = []
        self.covariances_ = []

class Stats(object):
    def __init__(self,path):
        if path != '' and path[-1] != '/':
            path += '/'
        file = open(path+'component_stats.txt')
        f = file.read()
        file.close()
        f = f.strip().splitlines()
        self.stats = [line.split('\t')[0] for line in f]
        self.stat2score = {s:-998 for s in self.stats}
    def set_stat(self,stat,value):
        self.stat2score[stat] = value


class AODE(object):
    '''
    Given a directory containing AODE parameters generated by SWIFr_train.py, compute
    posterior probabilities for each classification scenario. Can be run site-by-site
    using --interactive, or can be computed for every SNP in a file using --file. One 
    or the other flag must be provided.
    '''

    def __init__(self,path2trained):

        if path2trained != '' and path2trained[-1] != '/':
            path2trained += '/'
        self.path2trained = path2trained

        file = open(self.path2trained+'component_stats.txt','r')
        f = file.read()
        file.close()
        f = f.strip().splitlines()
        f = [x.strip() for x in f]
        self.statlist = f

        self.num2stat = {i:self.statlist[i] for i in range(len(self.statlist))}
        self.stat2num = {y:x for x,y in list(self.num2stat.items())}

        self.path2AODE = self.path2trained+'AODE_params/'    

        file = open(self.path2trained+'classes.txt','r')
        f = file.read()
        file.close()
        self.scenarios = [x.strip() for x in f.strip().splitlines()]
        print('classes:')
        print(self.scenarios)



        self.JOINTS = [[[] for stat2 in list(self.stat2num.keys())] for stat1 in list(self.stat2num.keys())]
        self.MARGINALS = [[] for stat in list(self.stat2num.keys())]

        for stat1 in self.statlist:
            for stat2 in [x for x in self.statlist if x!= stat1]:
                statnum1 = self.stat2num[stat1]
                statnum2 = self.stat2num[stat2]
                if statnum1 < statnum2:
                    for scenario in self.scenarios:
                        self.JOINTS[statnum1][statnum2].append(self.gmm_fit(stat1,stat2,scenario))

        for stat1 in self.statlist:
            statnum1 = self.stat2num[stat1]
            for scenario in self.scenarios:
                self.MARGINALS[statnum1].append(self.gmm_fit_1D(stat1,scenario))

    
    def aode(self,stats,pi):
        #pi must have length len(scenarios)
        numerators = [[] for i in range(len(self.scenarios))]
        raw_nums = [[] for i in range(len(self.scenarios))]
        denominators = []
        ode_num_dicts = {}
        for i in range(len(self.scenarios)):
            ode_num_dicts[self.scenarios[i]]=dict()
        for keystat in self.statlist:
            ps,nums,denom,raw = self.ode(keystat,stats,pi)
            for i in range(len(self.scenarios)):
                if ps != 'n/a':
                    ode_num_dicts[self.scenarios[i]][keystat]=nums[i]
                    numerators[i].append(nums[i])
                    raw_nums[i].append(raw[i])
                else:
                    ode_num_dicts[self.scenarios[i]][keystat]='-998'
            denominators.append(denom)
        aoderaw = [sum(raw_nums[i]) for i in range(len(self.scenarios))]
        if sum(denominators) > 0:
            aodeposteriors = [float(sum(numerators[i]))/sum(denominators) for i in range(len(self.scenarios))]
            return aodeposteriors, aoderaw, ode_num_dicts
        else:
            return [0 for i in range(len(self.scenarios))], aoderaw, ode_num_dicts

    def ode(self,keystat,stats,pi):
        score = stats.stat2score[keystat]
        if score == -998:
            return 'n/a',0,0,0
        else:
            MARGS = self.MARGINALS[self.stat2num[keystat]]
            Likelihoods = []
            for i in range(len(self.scenarios)):
                M = MARGS[i]
                Likelihoods.append(self.GMM_pdf(M,score))

            for stat in self.statlist:
                if stat != keystat:
                    score2 = stats.stat2score[stat]
                    if score2 != -998:
                        if self.stat2num[keystat] < self.stat2num[stat]:
                            for i in range(len(self.scenarios)):
                                H = self.conditional_GMM(score,2,self.JOINTS[self.stat2num[keystat]][self.stat2num[stat]][i])
                                Likelihoods[i] = Likelihoods[i]*self.GMM_pdf(H,score2)
                        else:
                            for i in range(len(self.scenarios)):
                                H = self.conditional_GMM(score,1,self.JOINTS[self.stat2num[stat]][self.stat2num[keystat]][i])
                                Likelihoods[i] = Likelihoods[i]*self.GMM_pdf(H,score2)
            numerators = [0 for i in range(len(self.scenarios))]
            for i in range(len(self.scenarios)):
                numerators[i] = float(pi[i]*Likelihoods[i])
            even_raw_aode = [0 for i in range(len(self.scenarios))]
            for i in range(len(self.scenarios)):
                even_raw_aode[i] = float(0.5*Likelihoods[i])
            denominator = sum(numerators)
            if denominator == 0:                
                return [0 for i in range(len(numerators))],numerators,denominator, Likelihoods
            else:
                posteriors = [float(numerators[i])/denominator for i in range(len(numerators))]
                return posteriors, numerators, denominator, even_raw_aode


    
    def GMM_pdf(self,G,x):
        w = G.weights_
        mu = G.means_
        C = G.covariances_
        pdf = 0
        for i in range(len(w)):
            pdf += w[i]*self.normpdf(x,mu[i][0],math.sqrt(C[i][0]))
        return pdf


    def normpdf(self,x,mu,sigma):
        u = float(x-mu)/sigma
        y = (old_div(1,(math.sqrt(2*math.pi)*abs(sigma))))*math.exp(old_div(-u*u,2))
        return y

    def calc_srs(self,raw_probs):
        log_probs=[]
        for i in raw_probs:
            if i>0:
                log_probs.append(math.log10(i))
            else:
                log_probs.append(-inf)
        return max(log_probs)
    
    def conditional_GMM(self,condval,keystat,G):
        #keystat = 1 if want stat1|stat2, keystat = 2 if want stat2|stat1
        H = Mix1D()

        for i in range(len(G.weights_)):
            sigma1 = math.sqrt(G.covariances_[i][0][0])
            sigma2 = math.sqrt(G.covariances_[i][1][1])
            ro = float(G.covariances_[i][0][1])/(sigma1*sigma2)
            mu1 = G.means_[i][0]
            mu2 = G.means_[i][1]
            if keystat == 1:
                H.weights_.append(G.weights_[i])
                H.means_.append([mu1 + float(sigma1*ro*(condval-mu2))/sigma2])
                H.covariances_.append([(1-ro**2)*sigma1**2])

            elif keystat == 2:
                H.weights_.append(G.weights_[i])
                H.means_.append([mu2 + float(sigma2*ro*(condval-mu1))/sigma1])
                H.covariances_.append([(1-ro**2)*sigma2**2])
        return H

    def gmm_fit(self,stat1,stat2,scenario):
        sys.path.append(os.getcwd())
        G = pickle.load(open(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_GMMparams.p','rb'))
        return G

    def gmm_fit_1D(self,stat,scenario):
        G = pickle.load(open(self.path2AODE+stat+'_'+scenario+'_1D_GMMparams.p','rb'))
        return G

    def calculate_on_file(self,filename,pivec,outfile,A,nb,ode_diagnostics):
        file = open(filename,'r')
        f = file.read()
        file.close()
        out = open(outfile,'w')
        #out = open(filename.replace('.txt','')+'_classified','w')
        f = f.strip().splitlines()
        header = f[0]
        H = header.strip().split('\t')
        newheader = header+'\t'
        for s in self.scenarios:
            newheader += 'pi_'+s+'\t'
        for s in self.scenarios:
            newheader += 'P('+s+')'+'\t'
        newheader += 'SRS'+'\t'
        if ode_diagnostics:
            for s in self.scenarios:
                newheader += 'RawAODE_P('+s+')'+'\t'
                for stat in self.statlist:
                    newheader += stat + '_ODE('+s+')' + '\t'
        if nb:
            for s in self.scenarios:
                newheader += 'NB_P('+s+')'+'\t'
        newheader = newheader.strip()
        out.write(newheader+'\n')
        stat2index = {}
        for stat in self.statlist:
            stat2index[stat] = H.index(stat)
        f = f[1:]
        for line in f:
            L = line.strip().split('\t')
            S = Stats(self.path2trained)
            for stat in S.stats:
                S.stat2score[stat] = float(L[stat2index[stat]])
            scenario_probs, raw_probs, odes = A.aode(S,pivec)
            outtext = line+'\t'
            for i in range(len(pivec)):
                outtext += str(pivec[i])+'\t'
            for i in range(len(self.scenarios)):
                outtext += str(scenario_probs[i])+'\t'
            outtext += str(A.calc_srs(raw_probs))+'\t'
            if ode_diagnostics:
                for i in range(len(self.scenarios)):
                    outtext += str(raw_probs[i])+'\t'
                    for stat in self.statlist:
                        outtext += str(odes[self.scenarios[i]][stat])+'\t'
            if nb:
                nb_scores, nums, denoms = A.naive_bayes(S,pivec)
                for i in range(len(self.scenarios)):
                    outtext += str(nb_scores[i])+'\t'
            outtext = outtext.strip()
            out.write(outtext+'\n')
        out.close()


    def naive_bayes(self,stats,pi):
        #pi must have length len(scenarios)
        Likelihoods = [1 for i in range(len(self.scenarios))] #product of P(statistic value | scenario) (for each scenario, a product across all statistics)
        for stat in self.statlist:
            score = stats.stat2score[stat] #value of statistic
            MARGS = self.MARGINALS[self.stat2num[stat]] #1-D distributions learned from SWIFr
            for i in range(len(self.scenarios)):
                M = MARGS[i] #distribution for scenario and stat
                Likelihoods[i] = Likelihoods[i]*self.GMM_pdf(M,score)
        numerators = [pi[i]*Likelihoods[i] for i in range(len(self.scenarios))] #multiply by prior scenario probs
        denominator = sum(numerators) #normalization
        # print(posteriors)
        # print(numerators)
        # print(denominator)
        if denominator == 0:
            return [0 for i in range(len(numerators))],numerators, denominator
        else:
            posteriors = [float(numerators[i])/denominator for i in range(len(numerators))]
            return posteriors, numerators, denominator




# if __name__ == '__main__':
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path2trained',action='store',dest='path2trained',default='')
    parser.add_argument('--interactive',action='store_true',dest='interactive',default=False)
    parser.add_argument('--file',action='store',dest='filename') #use instead of interactive mode to work on a whole file
    parser.add_argument('--pi',action='store',nargs='+',default=['0.99999','0.00001']) #can use with either mode
    parser.add_argument('--outfile',action='store',default='')
    parser.add_argument('--nb',action='store_true',dest='nb',default=False) #run naive bayes
    parser.add_argument('--ode',action='store_true',dest='ode',default=False) #output non-normalized AODE output and inidividual ODE scores
    args = parser.parse_args()
     
    if not args.interactive and not args.filename:
        print("Error: SWIF(r) must be run either with an input file using --file or in " \
             + "interactive mode using --interactive.")
        print("Please try running the program again.")
    
    else:
        A = AODE(args.path2trained)
    
        pivec = [float(x) for x in args.pi]
    
        if round(sum(pivec),10) != round(1,10):
            print('Error: prior values should add to 1')
        else:
            if len(pivec) != len(A.scenarios):
                print('Error: need '+str(len(A.scenarios))+' values for pi')
    
            else:
                print('Using priors:')
                for i in range(len(A.scenarios)):
                    print('p('+A.scenarios[i]+') = '+str(pivec[i]))
                if args.interactive:
                    S = Stats(args.path2trained)
                    for stat in S.stats:
                        response = eval(input("Value for "+stat+": "))
                        S.stat2score[stat] = float(response)
                
                    scenario_probs = A.aode(S,pivec)
                    for i in range(len(A.scenarios)):
                        print('Probability of '+A.scenarios[i]+': '+str(scenario_probs[i]))
                else:
                    if args.outfile == '':
                        args.outfile = args.filename.replace('.txt','')+'_classified'
                    A.calculate_on_file(args.filename,pivec,args.outfile, A, args.nb, args.ode)


