import time

i = 999999999
t0 = time.time()
while i:
    i = i-1
print("Tiempo transcurrido=", time.time()-t0)

i = 999999999
t0 = time.time()
while i > 0:
    i = i-1
print("Tiempo transcurrido=", time.time()-t0)