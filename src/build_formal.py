import pickle, os, json, random
# Input corpus directories and output location, overridable from the environment.
MR = os.environ.get("MIX_DIR", "./mix")
S19 = os.environ.get("S19_DIR", "./s19")
OUT = os.environ.get("OUT_DIR", "./out")
BEST_ARM = os.environ.get("BEST_ARM_FILE", "/tmp/best_arm.txt")
best = open(BEST_ARM).read().strip()
ARMS = {
 "B0":   dict(code=.40, book=.25, math=.15, wiki=.10, lit=.08, zh=.02),
 "code+":dict(code=.50, book=.20, math=.14, wiki=.08, lit=.06, zh=.02),
 "code-":dict(code=.30, book=.29, math=.15, wiki=.12, lit=.10, zh=.04),
 "book+":dict(code=.35, book=.35, math=.14, wiki=.08, lit=.06, zh=.02),
 "math+":dict(code=.37, book=.22, math=.20, wiki=.10, lit=.08, zh=.03),
 "zh+":  dict(code=.38, book=.24, math=.14, wiki=.10, lit=.07, zh=.07),
 "wiki+":dict(code=.37, book=.22, math=.14, wiki=.16, lit=.08, zh=.03),
}
mix=ARMS[best]
pools={}
for dom,f in [("code",MR+"/poolbig_code.pkl"),("book",MR+"/poolbig_book.pkl"),
              ("math",MR+"/poolbig_math.pkl"),("lit",MR+"/poolbig_lit.pkl"),
              ("zh",S19+"/pool_zh.pkl"),("wiki",S19+"/pool_wiki.pkl")]:
    with open(f,"rb") as fh: pools[dom]=pickle.load(fh)
    print(dom, sum(len(a) for a in pools[dom]["train"]), flush=True)
rng=random.Random(42)
TOTAL=1_300_000_000
outp=os.path.join(OUT,"formal_cyano.bin")
stats={}
with open(outp+".tmp","wb") as fo:
    buf=[]
    for dom,share in mix.items():
        p=pools[dom]["train"]; quota=int(TOTAL*share); n=0
        # quality-weighted order for code, shuffled for others; cycle until quota
        if dom=="code":
            w=pools["w"] if False else None
            order=sorted(range(len(p)),key=lambda i:-(max(rng.random(),1e-12)**(1.0/max(pools[dom]["w"][i]**4,1e-6))))
        else:
            order=list(range(len(p))); rng.shuffle(order)
        oi=0
        while n<quota:
            if oi>=len(order): oi=0; rng.shuffle(order)
            a=p[order[oi]]; oi+=1
            buf.append(a); n+=len(a)
        stats[dom]=n
        print("dom",dom,n,flush=True)
    rng.shuffle(buf)
    for a in buf: a.tofile(fo)
os.replace(outp+".tmp",outp)
json.dump({"best":best,"mix":mix,"tokens":stats,"total":sum(stats.values())},open(OUT+"/formal_manifest.json","w"),indent=1)
print("DONE formal",flush=True)