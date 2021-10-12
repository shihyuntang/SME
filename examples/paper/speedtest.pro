; Run the batch file with idl -e ".r speedtest"
Set_Plot, "NULL"
restore,'/home/ansgar/Documents/Python/sme/examples/paper/speedtest.inp'
; Run once to load data files
starttime = SYSTIME(/SECONDS)
sme_main, sme
endtime = SYSTIME(/SECONDS)
runtime = endtime - starttime
print, "Runtime: ", runtime


totaltime = 0
times = []
for i = 1,1000 do begin
    starttime = SYSTIME(/SECONDS)
    sme_main, sme
    endtime = SYSTIME(/SECONDS)
    runtime = endtime - starttime
    print, runtime
    totaltime = totaltime + runtime
    times = [runtime, times]
endfor

totaltime = totaltime / 1000
print, "Runtime: ", totaltime, " s +- ", stddev(times)
print, "Runtime: ", min(times)


save, sme, file='/home/ansgar/Documents/Python/sme/examples/paper/speedtest.out'
end
