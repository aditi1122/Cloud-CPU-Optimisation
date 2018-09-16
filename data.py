import psutil
from datetime import date
import calendar
import datetime



try:
    f=open("set.csv",'w')
    f.write("Day,"+"Hour,"+"Cores"+"," +"Total"+","+"CPU1"+","+"CPU2"+","+"CPU3"+","+"CPU4"+","+"Actual Required"+'\n')
    n=4

    def actual_calculator(a):
        t=sum(a)/4
        if(t<30):
            return 1
        elif (t<50 and t>30):
            return 2
        elif (t<70 and t>50):
            return 3
        else:
            return 4
    
    while True:
        my_date = date.today()
        day=calendar.day_name[my_date.weekday()]
        now = datetime.datetime.now()
        a = psutil.cpu_percent(interval=0.5,percpu=True)
        act=actual_calculator(a)    
        f.write(day +','+str(now.hour)+','+str(n) +','+str(sum(a)/4) +','+str(a[0])+','+str(a[1])+','+str(a[2])+','+str(a[3])+','+str(act)+'\n' )

except:
    f.close()
    

diwakar shukla
hello how are you

    

