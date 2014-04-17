import csv

def readWithHeader( infile, delim="," ):
    with open( infile, 'rb') as csvFile:
        csvReader = csv.reader( csvFile, delimiter=delim )
        line = csvReader.next()
        fieldName = []
        fieldType = []
        header = []
        for l in line:
            #print l
            arr = l.strip().split(':')
            fieldName.append(arr[0])
            fieldType.append(arr[1])
            header.append(tuple(arr))
        #print fieldName
        #print fieldType
        #print header
            
        cont = []
        for obj in csvReader:
            #print len(obj), obj
            ll = []
            for i in range(0, len(fieldName)):
                #print "\t",obj[i]
                if fieldType[i]=="float":
                    ll.append(float(obj[i]))
                elif fieldType[i]=="int":
                    ll.append(int(obj[i]))
                elif fieldType[i]=="bool":
                    ll.append(bool(obj[i]))
                else:
                    ll.append(obj[i])
            #print dic
            cont.append(ll)
    
    #print cont
    return header, cont
