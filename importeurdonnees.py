import glob

def getin(nbfiles):
    allfiles = glob.glob(r"C:\Users\etien\Documents\DeblecxAI\Python\Traducteur\Données\In\*")

    allfiles = allfiles[:nbfiles]

    allIn = []

    for myfile in allfiles:

        myfile = open(myfile, "r")
        content = myfile.read()

        allIn.append(content)

        myfile.close()
    
    return allIn

def getout(nbfiles):
    allfiles = glob.glob(r"C:\Users\etien\Documents\DeblecxAI\Python\Traducteur\Données\Out\*")

    allfiles = allfiles[:nbfiles]

    allOut = []

    for myfile in allfiles:

        myfile = open(myfile, "r")
        content = myfile.read()

        allOut.append(content)

        myfile.close()
    
    return allOut