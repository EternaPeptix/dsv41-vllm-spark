import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
q = ("select s.value, k.gridX, k.blockX, k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory, "
     "count(*), avg(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
     "where s.value like ? or s.value like ? or s.value like ? or s.value like ? "
     "group by 1,2,3,4,5,6 order by 7 desc limit 25")
for r in c.execute(q, ("%skinny%", "%Kernel2%", "%MXFP8Quant%", "%gemvNSP%")):
    print(r)
