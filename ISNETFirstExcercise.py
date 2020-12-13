import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
Pi=np.pi


#I have desing the graphic part of this code to make the plots for the default values. If these values are changed the plotting part of the code might get weird, but the numerical results should be fine and they will be printed into .txt files, so you can go and take a look there if the graphs are nonsense.

#In this particular exercise we don't care much about the bias, we are just focusing on the variance. This is why I don't define quantities like the \etas, the optimal parameters, or the irreducible error



##################################
###########Quantities the user might want to play with
##################################

#Here are the points where we are "measuring". You can create your own list. Also, the errors are the same for all but that can be eddited too!
qlocations=np.array([0.5,0.8,1.2])

#Error bar size for the measured points
sigmaErr=0.005


##################################
###########Technical definitions and details
##################################
#Scale to divide the errors by when calculating H1. That way the inverse matrix is easier to compute. 
ErrScale=0.001

#Scale to divide the "errors" when fitting the central parameters (they use the entire data)
ErrScaleFF=0.001

#Gaussian distribution for plotting gaussians
def gaussian(x, mu, sig):
    return 1.0/np.sqrt(2*Pi*sig**2)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


#Function I created to convert a float to a string in scientific notation with a defined precision. We will use it when writing the results into the .txt files
def ScNotListToString(ListVals,pres):
    ListIn=[]
    for kIn in range(len(ListVals)):
        formin='{:.'+str(pres)+'e}'
        ListIn.append(formin.format(ListVals[kIn]))
    return str(    "["+",".join(map(str,ListIn)) +"]"    )



##################################
###########Data prepatation
##################################

#Error (sigma) as a function of q
def ErrorQ(q):
    return sigmaErr

#Function that takes a list of qs and returns a list of [q, Error(q)]
def ErrorQList(qListE):
    EQLIn=np.array([])
    for iE in range(len(qListE)):
        EQLIn=np.append(EQLIn,ErrorQ(qListE[iE]))
    return EQLIn

#Data importing and truth definition:

data_path = 'Data/'

ExpChFF = np.genfromtxt(data_path+'ChFFPb208.csv', delimiter=',')

#Here we are importing the charge data values, we need them to calculate the radius. Also, we might at some point want to see how the model reproduces the density
ExpChRho = np.genfromtxt(data_path+'ChDensityPb208.csv', delimiter=',')


#We interpolate between the true data values using a cubic spling
TruthFSpline=CubicSpline(ExpChFF[:,0],ExpChFF[:,1])

TruthRhoSpline=CubicSpline(ExpChRho[:,0],ExpChRho[:,1])

#This is the truth generator for the q-space
def TruthF(q):
    return TruthFSpline(q)
#This is the truth generator for the rho-space
def TruthRho(r):
    return TruthRhoSpline(r)


#Function that takes a list of qs and returns a list of [q, Ftrue(q)]
def DataMaker(qListIn):
    return (np.array((np.array(qListIn) , np.array(TruthF(qListIn))   )).T)


#Here are the points where we are "measuring", we create the associated "y" values and their errors
data=DataMaker(qlocations)
errors=ErrorQList(qlocations)


##################################
###########Model Information
##################################


#Here we import the information about the model
from Models.SFModel import *

#Main function that fits the parameters
def ParamFitter(Data,Errors,P0):
    popt, pcov =curve_fit(FNoList, Data.T[0],Data.T[1],p0=P0,sigma=Errors)
    return popt
    

CentParam=ParamFitter(ExpChFF,[ErrScaleFF]*(len(ExpChFF)),ParS0)


#This function calculates the true radius from the experimental Rho data
def RCalculator(RhoPoints):
    Rin=0
    for i in range(len(ExpChRho)-1):
        Rin=Rin+ExpChRho[i][1]*(ExpChRho[i+1][0]-ExpChRho[i][0])*ExpChRho[i][0]**4
    return np.sqrt(4*Pi*Rin)

TruthRadius=RCalculator(ExpChRho)

#Function that retunrs the model radius
def MR(FParams):
    return ModelRadius(FParams)
#print(RCalculator(TruthRhoPoints,0))



##################################
###########Transfer Function Stuff
##################################


#This is the inverse Hessian Matrix for chi2. (By the way, I define my \chi2 with a 1/2 extra, so \chi2 is everything that sits in the exponential. This makes the Hessian matrix to lack its original extra 1/2. All the calculations are the same, is just a convention)
def H1(qDataH1,errListH1,qGradlistH1,paramlistH1):
    
    
    NyIn=len(qDataH1)
    
    H1In=np.zeros((Pz,Pz))
    
    qPredH1=np.array((np.array(qDataH1[:,0]),FList(qDataH1[:,0],paramlistH1))).T
    
    #We divide the errors by the Error Scale so the matrix we are going to invert has their entries more managables for inversion. At the end we multiply by that same scale so everything stays the same.
    errListIn=np.array(errListH1)*1.0/ErrScale
    
    for iH1 in range(Pz):
        for kH1 in range(iH1,Pz):
            for jH1 in range(NyIn):
                
                H1In[iH1][kH1]=H1In[iH1][kH1]+((qGradlistH1[jH1][iH1]*qGradlistH1[jH1][kH1]) -(qDataH1[jH1][1]-qPredH1[jH1][1])*(HessF(qDataH1[jH1][0],paramlistH1, iH1,kH1)))*1.0/(errListIn[jH1]**2)
            #H is symmetric, so no need to re calculate the other half: 
            H1In[kH1][iH1]= H1In[iH1][kH1]
            
    return np.linalg.inv(H1In)*(ErrScale**2)



#Function that creates the transfer functions of the parameters. The convention is that the first index is the parameter and the second the observation. For example, ParamTranFunc[1][3] describes how the second parameter is affected by changes in the fourth observation (remember that for python the lists starts at 0). 
def ParamTranFunc(qDataPTF,errListPTF,paramlistPTF):
    gradFPTFin=GradFList(qDataPTF[:,0],paramlistPTF)
    matH1PTFin=H1(qDataPTF,errListPTF,gradFPTFin,paramlistPTF)

    for iPTF in range(len(gradFPTFin)):
        gradFPTFin[iPTF,:]*=(errListPTF[iPTF]**(-2))

    return np.matmul(matH1PTFin,(gradFPTFin.T))




#Function that creates the transfer functions of the Radius. It has only one index and it describes how that observation affects the estimated radius.
#(If you are reading this comment you get a free cookie with your coffee or a pretzel with your beer when you cash in your coupon. It means a lot to me that you are taking such an effort to understand this code and the TFs ideas) 
def MRTranFunc(FParams,ParamTFIn):
    gradRIn=np.array([])
    for iM in range(Pz):
        gradRIn=np.append(gradRIn,GradRadius(FParams,iM))
    return np.matmul(gradRIn,ParamTFIn)



#This function calculates the extracted radius and its variance as a funciton of the qlocations
def RadiusAndVariance(qvaluesrad,errorsrad):
    dataRad=DataMaker(qvaluesrad)
    paramRad=ParamFitter(dataRad,errorsrad,ParS0)
    TFParamRad=ParamTranFunc(dataRad,errorsrad,paramRad)
    TFRad=MRTranFunc(paramRad,TFParamRad)
    varRad=0
    for j in range(len(dataRad)):
        varRad=varRad+(TFRad[j]*errorsrad[j])**2
    return [MR(paramRad),np.sqrt(varRad)]




##################################
###########Results Calculation
##################################


#Parameters for the data
param0=ParamFitter(data,errors,ParS0)

#Transfer functions calculation
TFParamsValues=ParamTranFunc(data,errors,param0)
TFRad=MRTranFunc(param0,TFParamsValues)









##################################
###########Plotting Stuff
##################################


def gaussian(x, mu, sig):
    return 1.0/np.sqrt(2*Pi*sig**2)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def GaussPlotter(FIGNAME,muG,sigG,trueval,colorvalue):
    x_values = np.linspace(min(muG-4*sigG,trueval-4*sigG), max(muG+4*sigG,trueval+4*sigG), 120)
    ys=gaussian(x_values, muG, sigG)
    FIGNAME.plot(x_values, gaussian(x_values, muG, sigG),linestyle='-',linewidth=4,color=colorvalue)
    FIGNAME.fill_between(x_values, gaussian(x_values, muG, sigG),alpha=0.3)
    FIGNAME.tick_params(axis='both', which='major', labelsize=20)
    plt.axvline(trueval,0.05,0.95,color='tab:pink',linewidth=6)



##################################
###########First plots with original error distribution
##################################


#First plots with original error distribution

fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
ax1=fig.add_subplot(211)
ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)

#We are plotting the errors as 20*Sigma so they can be seen.
plt.errorbar(data.T[0],data.T[1], yerr=errors*20,marker='o',color='red',markersize=10,linestyle='none')
ax1.tick_params(axis='both', which='major', labelsize=15)



#Radius transfer function values times their respective error added to the plot. If you change the default values you should disable these lines, otherwise the graphical interface will get weird
ax1.text(0.2, 0.15, abs(round(TFRad[0]*errors[0],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})

ax1.text(0.8, 0.1, abs(round(TFRad[1]*errors[1],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})

ax1.text(1.2, 0.15, abs(round(TFRad[2]*errors[2],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})




#Handmade legend
ax1.add_patch(patches.Rectangle((0.5,0.52),1.2,0.5,linewidth=2,edgecolor='k',facecolor='none'))
ax1.text(0.6, 0.85, '$TF_j \  \sigma_j$', style='italic', fontsize=20,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10,'edgecolor': 'black','linewidth':2})
ax1.text(0.95, 0.8, 'Transfer function\n times error [fm]', fontsize=15,
        )

plt.errorbar([0.7],[0.63], yerr=[0.1],marker='o',color='red',markersize=10,linestyle='none')

ax1.text(0.9, 0.6, 'Data Point (x20 Sigma)', fontsize=15,
        )



#Labeling the axes 
ax1.set_xlabel('$q \  [fm^{-1}]$',fontsize=20)
ax1.set_ylabel('$ F(q) $',fontsize=20)
ax1.set_title('$^{208}$Pb Electric Form Factor',fontsize=25)





#Lower plot that shows the gaussian
ax2=fig.add_subplot(212)

#Labeling the axes
ax2.set_xlabel('Charge Radius R $[fm]$',fontsize=20)
ax2.set_ylabel('$ P(R) $',fontsize=20)
ax2.set_title('$^{208}$Pb Charge Radius',fontsize=25)


resultsRad=RadiusAndVariance(qlocations,errors)

GaussPlotter(ax2,resultsRad[0],resultsRad[1],TruthRadius,'blue')



#Handmade legend
pink_patch = patches.Patch(color='tab:pink', label='True Radius')
blue_patch = patches.Patch(color='blue', label='Extracted Radius #1')
ax2.legend(loc="upper left",handles=[pink_patch,blue_patch],fontsize=15)


ax2.set_xlim([5.25, 5.7])
ax2.set_ylim([-0.5, 15])


#Writing on the plot the variance on the radius
ax2.text(5.27, 8, '$\Delta R=$%s  fm'%(round(resultsRad[1],3)), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'blue','linewidth':4})

print("\n\n-------------------------------------\nWeb browser version: Please open FIG1-OriginalDataSet.png \n-------------------------------------\n\n")

print("We can reduce one of the error bars by a factor of two\n")

print("Which point should we choose? Type 1, 2, or 3 to select the first, second or third point respectively\n")


plt.tight_layout()
plt.savefig('FIG1-OriginalDataSet.png', dpi=fig.dpi)
plt.show()


#User interaction
pointchosen=int(input())
errors2=ErrorQList(qlocations)

##################################
###########Second round of plots with new errors
##################################

#We reduce by half one of the point's errors
for ij in range(len(qlocations)):
    if pointchosen==ij+1:
        errors2[ij]=errors2[ij]*1.0/2

#New parameters and TFs
param02=ParamFitter(data,errors2,ParS0)
TFParamsValues2=ParamTranFunc(data,errors2,param02)
TFRad2=MRTranFunc(param02,TFParamsValues2)



#First plot with the new error distribution
fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
ax1=fig.add_subplot(211)
ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)


#We are plotting the errors as 20*Sigma so they can be seen.
plt.errorbar(data.T[0],data.T[1], yerr=errors2*20,marker='o',color='red',markersize=10,linestyle='none')
ax1.tick_params(axis='both', which='major', labelsize=15)



#Radius transfer function values times their respective error added to the plot
ax1.text(0.2, 0.15, abs(round(TFRad2[0]*errors2[0],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})

ax1.text(0.8, 0.1, abs(round(TFRad2[1]*errors2[1],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})

ax1.text(1.2, 0.15, abs(round(TFRad2[2]*errors2[2],3)), style='italic', fontsize=15,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})


#Handmade legend
ax1.add_patch(patches.Rectangle((0.5,0.52),1.2,0.5,linewidth=2,edgecolor='k',facecolor='none'))
ax1.text(0.6, 0.85, '$TF_j \  \sigma_j$', style='italic', fontsize=20,
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10,'edgecolor': 'black','linewidth':2})
ax1.text(0.95, 0.8, 'Transfer function\n times error [fm]', fontsize=15,
        )
plt.errorbar([0.7],[0.63], yerr=[0.1],marker='o',color='red',markersize=10,linestyle='none')
ax1.text(0.9, 0.6, 'Data Point (x20 Sigma)', fontsize=15,
        )


#First plots with original error distribution
ax1.set_xlabel('$q \  [fm^{-1}]$',fontsize=20)
ax1.set_ylabel('$ F(q) $',fontsize=20)
ax1.set_title('$^{208}$Pb Electric Form Factor',fontsize=25)





#Lower plot that shows the gaussian
ax2=fig.add_subplot(212)

#First plots with original error distribution
ax2.set_xlabel('Charge Radius R $[fm]$',fontsize=20)
ax2.set_ylabel('$ P(R) $',fontsize=20)
ax2.set_title('$^{208}$Pb Charge Radius',fontsize=25)


resultsRad2=RadiusAndVariance(qlocations,errors2)
GaussPlotter(ax2,resultsRad[0],resultsRad[1],TruthRadius,'blue')
GaussPlotter(ax2,resultsRad2[0],resultsRad2[1],TruthRadius,'orange')


#Handmade legend
pink_patch = patches.Patch(color='tab:pink', label='True Radius')
blue_patch = patches.Patch(color='blue', label='Extracted Radius #1')
orange_patch = patches.Patch(color='orange', label='Extracted Radius #2')
ax2.legend(loc="upper left",handles=[pink_patch,blue_patch,orange_patch],fontsize=15)

ax2.text(5.27, 8, '$\Delta R=$%s  fm'%(round(resultsRad[1],3)), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'blue','linewidth':4})

ax2.text(5.27, 5, '$\Delta R=$%s  fm'%(round(resultsRad2[1],3)), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'orange','linewidth':4})


ax2.text(5.59, 10, 'Improvement', fontsize=15)
ax2.text(5.6, 8, '%s %%  '%int((-resultsRad2[1]+resultsRad[1])/resultsRad[1]*100), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'black','linewidth':4})

#Axes limits
ax2.set_xlim([5.25, 5.7])
ax2.set_ylim([-0.5, 15])

plt.tight_layout()
print("\n\n-------------------------------------\nWeb browser version: Please open FIG2-ReducedOneErrorBar.png \n-------------------------------------\n\n")
plt.savefig('FIG2-ReducedOneErrorBar.png', dpi=fig.dpi)
plt.show()
print("\n\n\nHere are the results for reducing your chosen error bar\n\n\n")
#dummystop=input()

##################################
###########Third round of plots for a new measurement
##################################


print("\n----------------------------")



print("We can now instead measure in a new location q4\n")

print("Lets make a plot of the error in the radius R as a function of this new q4 location\n")

print("\n\n-------------------------------------\nWeb browser version: Please open FIG3-SelectingFourthLoc.png \n-------------------------------------\n\n")

print("Where should we measure? Type the q value of your choice with two decimal places\n")


#This list will carry the information on how \Delta R changes as we place the new location q_4 at different places
deltaRValues=[]

for ij in range(len(ExpChFF)):
    deltaRValues.append([ExpChFF[ij][0],RadiusAndVariance(np.append(qlocations,ExpChFF[ij][0]),np.append(errors,[0.005]/np.sqrt(3)))[1]])



deltaRValues=np.array(deltaRValues)




#First plot with the original data
fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
ax1=fig.add_subplot(211)
ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)

#We are plotting the errors as 20*Sigma so they can be seen.
plt.errorbar(data.T[0],data.T[1], yerr=errors*20,marker='o',color='red',markersize=10,linestyle='none')

ax1.tick_params(axis='both', which='major', labelsize=15)


#Handmade legend
ax1.add_patch(patches.Rectangle((0.6,0.52),0.95,0.23,linewidth=2,edgecolor='k',facecolor='none'))
plt.errorbar([0.7],[0.63], yerr=[0.1],marker='o',color='red',markersize=10,linestyle='none')
ax1.text(0.8, 0.6, 'Data Point (x20 Sigma)', fontsize=15,
        )
ax1.set_xlabel('$q \  [fm^{-1}]$',fontsize=20)
ax1.set_ylabel('$ F(q) $',fontsize=20)
ax1.set_title('$^{208}$Pb Electric Form Factor',fontsize=25)


#Second plot that shows \Delta R as a function of q_4
ax2=fig.add_subplot(212)
ax2.plot(deltaRValues.T[0],deltaRValues.T[1],'b-',linewidth=4)
ax2.axhline(y=RadiusAndVariance(qlocations,errors)[1], color='r', linestyle='-',linewidth=4)

#Handmade legend
red_patch = patches.Patch(color='r', label='Original $\Delta R$')
blue_patch = patches.Patch(color='b', label='New $\Delta R$')
ax2.legend(loc="lower right",handles=[red_patch,blue_patch],fontsize=15)




#First plot with the new error distribution
ax2.set_xlabel('$q_4 \  [fm^{-1}]$',fontsize=20)
ax2.set_ylabel('$ \Delta R $ [fm]',fontsize=20)
ax2.set_title('$\Delta R$ as a function of the fourth location',fontsize=20)
ax2.tick_params(axis="x", labelsize=16) 
ax2.tick_params(axis="y", labelsize=16) 


plt.tight_layout()
plt.savefig('FIG3-SelectingFourthLoc.png', dpi=fig.dpi)
plt.show()

#The user selects the location
q4=float(input())

#We create the new dataset
qlocations2=np.append(qlocations,q4)
errors2=np.append(errors,0.005/np.sqrt(3))
data2=DataMaker(qlocations2)

#We create the new parameters and TFs
param02=ParamFitter(data2,errors2,ParS0)
TFParamsValues2=ParamTranFunc(data2,errors2,param02)
TFRad2=MRTranFunc(param02,TFParamsValues2)




##################################
###########Fourth round of plots with the new selected measurement
##################################


#First plot with the original data plus the new location
fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
ax1=fig.add_subplot(211)
ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)

#We are plotting the errors as 20*Sigma so they can be seen.
plt.errorbar(data.T[0],data.T[1], yerr=errors*20,marker='o',color='red',markersize=10,linestyle='none')
#Fourth location to be added now
plt.errorbar(data2[3][0],data2[3][1], yerr=errors2[3]*20,marker='o',color='blue',markersize=10,linestyle='none')

ax1.tick_params(axis='both', which='major', labelsize=15)







#Handmade legend
ax1.text(0.8, 0.63, 'Original Data (x20 Sigma)', fontsize=15,
        )
ax1.text(0.8, 0.85, 'New $q_4$ (x20 Sigma)', fontsize=15,
        )
ax1.add_patch(patches.Rectangle((0.6,0.5),1.05,0.5,linewidth=2,edgecolor='k',facecolor='none'))
plt.errorbar([0.7],[0.63], yerr=[0.1],marker='o',color='red',markersize=10,linestyle='none')
plt.errorbar([0.7],[0.85],yerr=[0.1],marker='o',color='blue',markersize=10,linestyle='none')


#Labeling the axes
ax1.set_xlabel('$q \  [fm^{-1}]$',fontsize=20)
ax1.set_ylabel('$ F(q) $',fontsize=20)
ax1.set_title('$^{208}$Pb Electric Form Factor',fontsize=25)



#Lower plot that shows the gaussians
ax2=fig.add_subplot(212)

#Labeling the axes
ax2.set_xlabel('Charge Radius R $[fm]$',fontsize=20)
ax2.set_ylabel('$ P(R) $',fontsize=20)
ax2.set_title('$^{208}$Pb Charge Radius',fontsize=25)

#Ploting the gaussians
resultsRad2=RadiusAndVariance(qlocations2,errors2)
GaussPlotter(ax2,resultsRad[0],resultsRad[1],TruthRadius,'blue')
GaussPlotter(ax2,resultsRad2[0],resultsRad2[1],TruthRadius,'orange')


#Handmade legend
pink_patch = patches.Patch(color='tab:pink', label='True Radius')
blue_patch = patches.Patch(color='blue', label='Extracted Radius #1')
orange_patch = patches.Patch(color='orange', label='Extracted Radius #2')
ax2.legend(loc="upper left",handles=[pink_patch,blue_patch,orange_patch],fontsize=15)

ax2.text(5.27, 10, '$\Delta R=$%s  fm'%(round(resultsRad[1],3)), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'blue','linewidth':4})
ax2.text(5.27, 5, '$\Delta R=$%s  fm'%(round(resultsRad2[1],3)), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'orange','linewidth':4})
ax2.text(5.59, 12, 'Improvement', fontsize=15)
ax2.text(5.6, 8, '%s %%  '%int((-resultsRad2[1]+resultsRad[1])/resultsRad[1]*100), fontsize=20 ,
        bbox={'facecolor': 'none', 'pad': 10,'edgecolor': 'black','linewidth':4})


ax2.set_xlim([5.25, 5.7])
ax2.set_ylim([-0.5, 20])

#Final plot call
plt.tight_layout()
print("\n\n-------------------------------------\nWeb browser version: Please open FIG4-SelectedNewLoc.png \n-------------------------------------\n\n")
plt.savefig('FIG4-SelectedNewLoc.png', dpi=fig.dpi)
plt.show()




