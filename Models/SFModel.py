import numpy as np
Pi=np.pi

#In these type of files (that end with the word "Model") we specify all the details needed for a model to be used by the program. In this file we are giving the details for a Symmetrized Fermi function of the form (Latex code): F(q,[a,c])=\frac{3}{qc[(qc)^2+(\pi qa)^2]}\Big[\frac{\pi qa}{\text{sinh}(\pi qa)}\Big]\Big[\frac{\pi qa}{\text{tanh}(\pi qa)}\text{sin}(qc)-qc\ \text{cos}(qc)  \Big]


ModelName=["SF"]

#Number of Parameters:
Pz=2

#Seed for the parameters:
ParS0=[0.5,6.6]


##Function F(variable; parameters adjustable by data):
##Par is = [a,c]
def F(q,Fpar):
    aux0=(Fpar[0]*(Pi*(((np.tanh((Fpar[0]*(Pi*q))))**(-1))*(np.sin((Fpar[1]*q))))))-(Fpar[1]*(np.cos((Fpar[1]*q))))
    aux1=((3.*(Fpar[0]*(Pi*(((np.sinh((Fpar[0]*(Pi*q))))**(-1))*aux0))))/q)/((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))
    return aux1/Fpar[1]


#I am adding this version of the function to use the python function fitter because I was having problems with minimizing \chi2 with the scipy minimizer. The function fitter likes the function to have the structure F(x, parameter 1, parameter 2, .... )
def FNoList(q,a,c):
    return F(q,[a,c])

#Function F(variable[LIST]; parameters adjustable by data). This function returns the data as a list
def FList(qlistF,Fpar):
    Fvalin=np.array([])
    for j in range(len(qlistF)):
        Fvalin=np.append(Fvalin,F(qlistF[j],Fpar))
    return Fvalin

#The gradient of the function that is modeling the form factor. The third argument is the parameter index (if we are taking the derivative with respect to the first parameter for example)
def GradF(q,Fpar,iIndex):
    if q==0:
        if iIndex==0:
            return 0
        
        if iIndex==1:
            return 0    
    
    else:
        if iIndex==0:
            aux0=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.tanh((Fpar[0]*(Pi*q))))**(-1))**2))*(np.sin((Fpar[1]*q)))))
            aux1=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.sinh((Fpar[0]*(Pi*q))))**(-1))**2))*(np.sin((Fpar[1]*q)))))
            aux2=(Fpar[1]*(((Fpar[1]**2)-((Fpar[0]**2)*(Pi**2)))*(np.cos((Fpar[1]*q)))))+(((Fpar[0]**2)*((Pi**2)*aux0))+((Fpar[0]**2)*((Pi**2)*aux1)))
            aux3=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q)))))+(2.*(Fpar[1]*(np.sin((Fpar[1]*q)))))
            aux4=((np.sinh((Fpar[0]*(Pi*q))))**(-1))*(aux2-(Fpar[0]*(Fpar[1]*(Pi*(((np.tanh((Fpar[0]*(Pi*q))))**(-1))*aux3)))))
            return ((-3.*(Pi*((((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-2.)*aux4)))/q)/Fpar[1]

        
        if iIndex==1:
            aux0=(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q))))))-(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sin((Fpar[1]*q))))
            aux1=(2.*(Fpar[1]*(np.cos((Fpar[1]*q)))))+(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.sin((Fpar[1]*q)))))
            aux2=((np.sinh((Fpar[0]*(Pi*q))))**(-1))*((Fpar[0]*(Pi*(((np.tanh((Fpar[0]*(Pi*q))))**(-1))*aux0)))+((Fpar[1]**2)*aux1))
            return (3.*(Fpar[0]*(Pi*((((Fpar[1]**3.)+((Fpar[0]**2)*(Fpar[1]*(Pi**2))))**-2.)*aux2))))/q


    
##GradFList gives back a list of the gradients of the model evaluated at the different datapoints    
def GradFList(qvals,Fpar):
    gradlistFinj=np.array([])
    for iG in range(len(Fpar)):
        gradlistFinj=np.append(gradlistFinj,GradF(qvals[0],Fpar,iG))
    gradlistFin=gradlistFinj
    for jG in range(1,len(qvals)):
        gradlistFinj=np.array([])
        for iG in range(len(Fpar)):
            gradlistFinj=np.append(gradlistFinj,GradF(qvals[jG],Fpar,iG))
        gradlistFin=np.vstack((gradlistFin,gradlistFinj))
    return gradlistFin


##Hessian Matrix of the MODEL (not \chi2). This is used to calculate the Hessian of \chi2 since there is a term there proportional to second derivatives of F. 
def HessF(q,Fpar, iIndex, kIndex):
    if q==0:
        if iIndex==0:
            
            if kIndex==0:
                return 0
            
            if kIndex==1:
                return 0

        if iIndex==1:
            
            if kIndex==0:
                return 0
            
            if kIndex==1:
                return 0
        
    else:
        if iIndex==0:
            if kIndex==0:
                aux0=((((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**2))*((q**2)*((((np.tanh((Fpar[0]*(Pi*q))))**(-1))**3.)*(np.sin((Fpar[1]*q)))))
                aux1=((((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**2))*((q**2)*(((((np.sinh((Fpar[0]*(Pi*q))))**(-1))**2))*(np.sin((Fpar[1]*q)))))
                aux2=(((Fpar[1]**4.)-((Fpar[0]**4.)*(Pi**4.)))*(q*(np.cos((Fpar[1]*q)))))+(Fpar[1]*(((Fpar[1]**2)+(-3.*((Fpar[0]**2)*(Pi**2))))*(np.sin((Fpar[1]*q)))))
                aux3=((np.tanh((Fpar[0]*(Pi*q))))**(-1))*((5.*((Fpar[0]**2)*((Pi**2)*aux1)))+(2.*(Fpar[1]*aux2)))
                aux4=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q)))))+(4.*(Fpar[1]*(np.sin((Fpar[1]*q)))))
                aux5=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.sinh((Fpar[0]*(Pi*q))))**(-1))**2))*aux4))
                aux6=Pi*((2.*(((-3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.cos((Fpar[1]*q)))))+aux5)
                aux7=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q)))))+(4.*(Fpar[1]*(np.sin((Fpar[1]*q)))))
                aux8=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.tanh((Fpar[0]*(Pi*q))))**(-1))**2))*aux7))
                aux9=((((Fpar[0]**2)*((Pi**2)*aux0))+aux3)-(Fpar[0]*(Fpar[1]*aux6)))-(Fpar[0]*(Fpar[1]*(Pi*aux8)))
                aux10=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-3.)*(((np.sinh((Fpar[0]*(Pi*q))))**(-1))*aux9)
                return ((3.*((Pi**2)*aux10))/q)/Fpar[1]

            
            if kIndex==1:
                aux0=((Fpar[0]**2)*((Pi**2)*(-2.+((Fpar[0]**2)*((Pi**2)*(q**2))))))+(2.*((Fpar[1]**2)*(3.+((Fpar[0]**2)*((Pi**2)*(q**2))))))
                aux1=(((Fpar[1]**4.)*(q**2))+aux0)*(((np.tanh((Fpar[0]*(Pi*q))))**(-1))*(np.sin((Fpar[1]*q))))
                aux2=(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q))))))-(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sin((Fpar[1]*q))))
                aux3=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.tanh((Fpar[0]*(Pi*q))))**(-1))**2))*aux2))
                aux4=(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q))))))-(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sin((Fpar[1]*q))))
                aux5=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.sinh((Fpar[0]*(Pi*q))))**(-1))**2))*aux4))
                aux6=(-2.*(Fpar[1]*(((Fpar[1]**2)+(-3.*((Fpar[0]**2)*(Pi**2))))*(np.cos((Fpar[1]*q))))))+((((Fpar[0]**4.)*(Pi**4.))-(Fpar[1]**4.))*(q*(np.sin((Fpar[1]*q)))))
                aux7=((Fpar[0]**2)*((Pi**2)*aux3))+(((Fpar[0]**2)*((Pi**2)*aux5))+((Fpar[1]**2)*aux6))
                aux8=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-3.)*(((np.sinh((Fpar[0]*(Pi*q))))**(-1))*((Fpar[0]*((Fpar[1]**2)*(Pi*aux1)))+aux7))
                return (-3.*((Fpar[1]**-2.)*(Pi*aux8)))/q
        if iIndex==1:
            if kIndex==0:
                aux0=((Fpar[0]**2)*((Pi**2)*(-2.+((Fpar[0]**2)*((Pi**2)*(q**2))))))+(2.*((Fpar[1]**2)*(3.+((Fpar[0]**2)*((Pi**2)*(q**2))))))
                aux1=(((Fpar[1]**4.)*(q**2))+aux0)*(((np.tanh((Fpar[0]*(Pi*q))))**(-1))*(np.sin((Fpar[1]*q))))
                aux2=(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q))))))-(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sin((Fpar[1]*q))))
                aux3=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.tanh((Fpar[0]*(Pi*q))))**(-1))**2))*aux2))
                aux4=(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.cos((Fpar[1]*q))))))-(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sin((Fpar[1]*q))))
                aux5=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(((((np.sinh((Fpar[0]*(Pi*q))))**(-1))**2))*aux4))
                aux6=(-2.*(Fpar[1]*(((Fpar[1]**2)+(-3.*((Fpar[0]**2)*(Pi**2))))*(np.cos((Fpar[1]*q))))))+((((Fpar[0]**4.)*(Pi**4.))-(Fpar[1]**4.))*(q*(np.sin((Fpar[1]*q)))))
                aux7=((Fpar[0]**2)*((Pi**2)*aux3))+(((Fpar[0]**2)*((Pi**2)*aux5))+((Fpar[1]**2)*aux6))
                aux8=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-3.)*(((np.sinh((Fpar[0]*(Pi*q))))**(-1))*((Fpar[0]*((Fpar[1]**2)*(Pi*aux1)))+aux7))
                return (-3.*((Fpar[1]**-2.)*(Pi*aux8)))/q

            
            if kIndex==1:
                aux0=(3.*(Fpar[1]**4.))+((4.*((Fpar[0]**2)*((Fpar[1]**2)*(Pi**2))))+((Fpar[0]**4.)*(Pi**4.)))
                aux1=(2.*((Fpar[1]**4.)*(-6.+((Fpar[0]**2)*((Pi**2)*(q**2))))))+((Fpar[0]**2)*((Fpar[1]**2)*((Pi**2)*(-6.+((Fpar[0]**2)*((Pi**2)*(q**2)))))))
                aux2=((-2.*((Fpar[0]**4.)*(Pi**4.)))+(((Fpar[1]**6.)*(q**2))+aux1))*(np.sin((Fpar[1]*q)))
                aux3=((np.tanh((Fpar[0]*(Pi*q))))**(-1))*((2.*(Fpar[1]*(aux0*(q*(np.cos((Fpar[1]*q)))))))+aux2)
                aux4=(2.*((Fpar[1]**2)*(-3.+((Fpar[0]**2)*((Pi**2)*(q**2))))))+((Fpar[0]**2)*((Pi**2)*(2.+((Fpar[0]**2)*((Pi**2)*(q**2))))))
                aux5=((((Fpar[1]**4.)*(q**2))+aux4)*(np.cos((Fpar[1]*q))))+(-4.*(Fpar[1]*(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*(q*(np.sin((Fpar[1]*q)))))))
                aux6=(((Fpar[1]**3.)+((Fpar[0]**2)*(Fpar[1]*(Pi**2))))**-3.)*(((np.sinh((Fpar[0]*(Pi*q))))**(-1))*((Fpar[0]*(Pi*aux3))-((Fpar[1]**3.)*aux5)))
                return (-3.*(Fpar[0]*(Pi*aux6)))/q



#Definitions of M (quantities of interest). M1 is the density Rho and M2 is the radius:

#M1 (RHO):________________________
def Rho(r,Fpar):
    if r==0:
        aux0=((0.75*(np.sinh((Fpar[1]/Fpar[0]))))/(1.+(np.cosh((Fpar[1]/Fpar[0])))))/((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))
        return(aux0/Pi)/Fpar[1]
    else:
        aux0=(0.75*(np.sinh((Fpar[1]/Fpar[0]))))/((np.cosh((Fpar[1]/Fpar[0])))+(np.cosh((r/Fpar[0]))))
        return((aux0/((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2))))/Pi)/Fpar[1]


#Function Rho(variable[LIST]; parameters adjustable by data). This function returns the data as a list
def RhoList(rList,Fpar):
    rlistin=np.array([])
    for iRL in range(len(rList)):
        rlistin=np.append(rlistin,Rho(rList[iRL],Fpar))
    return rlistin



#The gradient of the function that is modeling the density (we don't use this in the default exercise). The third argument is the parameter index (if we are taking the derivative with respect to the first parameter for example)
def GradRho(r,Fpar,iIndex):
    if r==0:
        if iIndex==0:
               aux0=((Fpar[0]**2)*(Fpar[1]*(Pi**2)))+(2.*((Fpar[0]**3.)*((Pi**2)*(np.sinh((Fpar[1]/Fpar[0]))))))
               aux1=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-2.)*(((((np.cosh(((0.5*Fpar[1])/Fpar[0])))**(-1))**2))*((Fpar[1]**3.)+aux0))
               return ((-0.375*((Fpar[0]**-2.)*aux1))/Pi)/Fpar[1]
        if iIndex==1:
               aux0=((Fpar[1]**3.)+((Fpar[0]**2)*(Fpar[1]*(Pi**2))))-(Fpar[0]*(((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*(np.sinh((Fpar[1]/Fpar[0])))))
               aux1=(((Fpar[1]**3.)+((Fpar[0]**2)*(Fpar[1]*(Pi**2))))**-2.)*(((((np.cosh(((0.5*Fpar[1])/Fpar[0])))**(-1))**2))*aux0)
               return ((0.375*aux1)/Pi)/Fpar[0]
       
    else:
        if iIndex==0:
            aux0=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*((np.cosh((Fpar[1]/Fpar[0])))*(np.cosh((r/Fpar[0]))))
            
            aux1=(2.*((Fpar[0]**3.)*((Pi**2)*((np.cosh((r/Fpar[0])))*(np.sinh((Fpar[1]/Fpar[0])))))))+((Fpar[0]**3.)*((Pi**2)*(np.sinh(((2.*Fpar[1])/Fpar[0])))))
            
            aux2=((Fpar[1]**3.)+(((Fpar[0]**2)*(Fpar[1]*(Pi**2)))+((Fpar[1]*aux0)+aux1)))-((Fpar[0]**2)*((Pi**2)*(r*((np.sinh((Fpar[1]/Fpar[0])))*(np.sinh((r/Fpar[0])))))))
            
            aux3=(((np.cosh((Fpar[1]/Fpar[0])))+(np.cosh((r/Fpar[0]))))**-2.)*(aux2-((Fpar[1]**2)*(r*((np.sinh((Fpar[1]/Fpar[0])))*(np.sinh((r/Fpar[0])))))))
            
            return ((-0.75*((Fpar[0]**-2.)*((((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-2.)*aux3)))/Pi)/Fpar[1]
        
        
        if iIndex==1:
            aux0=((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))*((np.cosh((Fpar[1]/Fpar[0])))*(np.cosh((r/Fpar[0]))))
            aux1=(-1.5*(Fpar[0]*((Fpar[1]**2)*(np.sinh(((2.*Fpar[1])/Fpar[0]))))))+(-0.5*((Fpar[0]**3.)*((Pi**2)*(np.sinh(((2.*Fpar[1])/Fpar[0]))))))
            aux2=((3.*(Fpar[1]**2))+((Fpar[0]**2)*(Pi**2)))*((np.cosh((r/Fpar[0])))*(np.sinh((Fpar[1]/Fpar[0]))))
            aux3=((Fpar[1]**3.)+(((Fpar[0]**2)*(Fpar[1]*(Pi**2)))+((Fpar[1]*aux0)+aux1)))-(Fpar[0]*aux2)
            aux4=(((Fpar[1]**2)+((Fpar[0]**2)*(Pi**2)))**-2.)*((((np.cosh((Fpar[1]/Fpar[0])))+(np.cosh((r/Fpar[0]))))**-2.)*aux3)
            return ((0.75*((Fpar[1]**-2.)*aux4))/Pi)/Fpar[0]
        
#Don't need to specify the index in this one, it returns the whole thing
def GradRhoVec(r,Fpar):
    GRhoVecIn=np.array([GradRho(r,Fpar,0)])
    for kg in range(1,Pz):
        GRhoVecIn=np.append(GRhoVecIn,GradRho(r,Fpar,kg))
    return GRhoVecIn


#M2 (R):________________________


#The radius as a function of the parameters
def ModelRadius(FPar):
    return np.sqrt(3.0/5.0*FPar[1]**2+7.0/5.0*Pi**2*FPar[0]**2)


#The gradient of the radius as a function of the parameters. The second argument is the parameter index (if we are taking the derivative with respect to the first parameter for example)
def GradRadius(Fpar,index):
    if index==0:
        return 7.0/5.0*Pi**2*Fpar[0]/ModelRadius(Fpar)
    if index==1:
        return 3.0/5.0*Fpar[1]/ModelRadius(Fpar)









