import psutil
from datetime import date
import calendar


f=open("set2.csv",'w')
f.write("Day,"+"Cores"+"," +"Total"+","+"CPU1"+","+"CPU2"+","+"CPU3"+","+"CPU4"+","+"Actual Required"+'\n')
n=4

def actual_calculator(a):
    if(sum(a)<30):
        return 1
    elif (sum(a)<50 and sum(a)>30):
        return 2
    elif (sum(a)<70 and sum(a)>50):
        return 3
    else:
        return 4
    
while True:
    my_date = date.today()
    day=calendar.day_name[my_date.weekday()]
    a = psutil.cpu_percent(interval=0.5,percpu=True)
    act=actual_calculator(a)
    f.write(day +','+str(n) +','+str(sum(a)) +','+str(a[0])+','+str(a[1])+','+str(a[2])+','+str(a[3])+','+str(act)+'\n' )
    
    
    
