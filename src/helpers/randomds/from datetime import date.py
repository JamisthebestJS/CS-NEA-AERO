from datetime import date
def valid_date(y, m ,d):
    try:
        date(y, m, d)
        return True
    except:
        return False

results = set()

for yy in range(100):
    for century in [1900, 2000, 2100]:
        year = century + yy
        for m in range(1, 13):
            for d in range(1, 32):
                if not valid_date(year, m ,d):
                    continue

                candidates = [
                    f"{d:02d}{m:02d}{yy:02d}", #DDMMYY
                    f"{m:02d}{d:02d}{yy:02d}", #MMDDYY
                    f"{yy:02d}{m:02d}{d:02d}", #YYMMDD
                ]

                for s in candidates:
                    if ("28" in [s[0:2], s[2:4], s[4:6]]) and ("28" not in [s[1:3], s[3:5]]) and\
                        not ( 31 < int(s[0:2]) <= 99 or 31 < int(s[2:4]) <= 99 or 31 < int(s[4:6]) <= 99 ) and\
                        not ("7" in s) and not ("3" in s):
                        count = 0
                        for i in range(6):
                            if s[i] == "2" and s[i-1] == "8":
                                count +=1
                            if s[i] == s[i-1] and i%2 == 1:
                                count = 8
                                break
                            if i>0 and i%2 == 0 and s[i-1:i] == s[i-3:i-1]:
                                count = 8
                                break
                        if count <2:
                            results.add(s)
                        
                    
                        


with open("28_date_combinations.txt", "w") as f:
    for r in results:
        f.write(r + "\n")

print(len(results))