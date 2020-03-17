import os
import sys, getopt
import SimpleFunctions as ISF

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "c:n:b",["ConstitutiveModel=", "NumberElements=", "BCsType="])
    except getopt.GetoptError:
        print("CallSimpleFunctions.py -c <Constitutive Model> -n <Number of Elements> -b <Type of Boundary Conditions")
        sys.exit(2)

    c=None
    n=None
    b=None
                                               
    for opt, arg in opts:
        if opt in ("-c", "--ConstitutiveModel"):
            c = arg
        if opt in ("-n", "--NumberElements"):
            n = arg
        if opt in ("-n", "--BCsType"):
            b = arg
            
    if None is not in [c, n, b]:                                               

        LoadCases = ['Compression', 'Tension', 'SimpleShear']
        RelativeWeights = [1,1,1]


        FinalRelativeStretch = 0.1
        RelativeStepSize     = 0.02


        ResultsOptimization = ISF.ParametersOptimization(c, int(n), b, LoadCases, RelativeWeights, FinalRelativeStretch, RelativeStepSize)


        print(CallSimpleFunctions Ended)
     
    else:
        print(c, n, b)
                                                   
if __name__ == "__main__":
    main(sys.argv[1:])