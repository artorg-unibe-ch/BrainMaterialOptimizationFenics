import os
import sys, getopt
import SimpleFunctions as SF

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "n:",["NumberElements="])
    except getopt.GetoptError:
        print("call-with-parameters.py -n <Number of Elements>")
        sys.exit(2)

    n=None
                                               
    for opt, arg in opts:
        if opt in ("-n", "--NumberElements"):
            n = arg                                           
    
    if n is not None:                                               

        # Fixed parameters
        ConstitutiveModels = ['Ogden', 'Neo-Hookean']
        ConstitutiveModel = ConstitutiveModels[1]

        LoadCases = ['Compression', 'Tension', 'SimpleShear']
        RelativeWeights = [1,1,1]


        FinalRelativeStretch = 0.1
        RelativeStepSize     = 0.02


        ResultsOptimization = SF.ParametersOptimization(ConstitutiveModel, int(n), LoadCases, RelativeWeights, FinalRelativeStretch, RelativeStepSize)


        print("Number of elements by edge '%i'"%(NumberElements))
     
    else:
        print(n)
                                                   
if __name__ == "__main__":
    main(sys.argv[1:])