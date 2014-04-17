import sys

connSep = "_"
inSep = "\t"
outSep = "\t"

def main( infile, joinCols, splitCols, resvCols ):
    with open(infile, 'r') as f:
        hdr = f.readline().strip().split(inSep)
        jnHdr = connSep.join([ hdr[i].split(":")[0] for i in joinCols ]) + ":string"
        resvHdr = [ hdr[i] for i in resvCols ]
        #print jnHdr
        #print resvHdr
        
        dic = {}
        for line in f:
            arr = line.strip().split(inSep)
            spkey = connSep.join([arr[i] for i in splitCols])
            jnkey = connSep.join([arr[i] for i in joinCols])
            val = [arr[i] for i in resvCols]
            
            dic.setdefault(jnkey, {})
            dic[jnkey].setdefault(spkey, val)
            #print jnkey, spkey
    
    # Sort keys of dic
    keys = dic.keys()
    keys.sort()
    
    # Get all split lables
    spHdrDic = {}
    for k in keys:
        for l in dic[k]:
            spHdrDic.setdefault(l, 1)
    spHdr = spHdrDic.keys()
    spHdr.sort()
    #print spHdr
    
    # Complete empty value
    emptVal = [ "0" for i in resvCols]
    for k in keys:
        for l in spHdr:
            dic[k].setdefault(l, emptVal)
    
    # Print new header
    print jnHdr + outSep + outSep.join([ s+connSep+t for s in spHdr for t in resvHdr])
    for k in keys:
        val = [ dic[k][i][j] for i in spHdr for j in range(0,len(resvCols))]
        #print val
        print k + outSep + outSep.join(val)
        #print [ dic[k][i] for i in dic[k] ]
    
    return 0
            
        
if __name__ == "__main__":
    infile = sys.argv[1]
    jnCols = [ int(i) for i in sys.argv[2].strip().split(",") ]
    spCols = [ int(i) for i in sys.argv[3].strip().split(",") ]
    resvCols = [ int(i) for i in sys.argv[4].strip().split(",") ]
    #print jnCols,spCols,resvCols
    sys.exit( main(infile, jnCols, spCols, resvCols) )
    