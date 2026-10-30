import json,sys
a=json.load(open(sys.argv[1]))["rows"]; b=json.load(open(sys.argv[2]))["rows"]
for x,y in zip(a,b):
    s,t=x["text"],y["text"]; n=next((i for i in range(min(len(s),len(t))) if s[i]!=t[i]),None)
    if n is None: print(x["i"],"IDENTICAL" if s==t else "prefix-equal len %d/%d"%(len(s),len(t))); continue
    print(x["i"],"div@char %d of %d/%d"%(n,len(s),len(t)),"reasoning" if "\x00" in s[n:] else "content", repr(s[max(0,n-30):n+25]),"|",repr(t[n:n+25]))
