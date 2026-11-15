>>> w.summary()
n                              5857505067604849028619727330686829482307
digits                                                               40
bits                                                                133
small factor (< 10,000)                                            None
perfect square                                                    False
probable prime                                                    False
base (primes)                                                       400
half interval M                                                   16384
large-prime multiple                                                 40
threshold slack                                                     0.7
extra relations                                                      32
double-large-prime exponent                                         1.8
workers                                                               1
sieve kernel                                                      numba
Name: input, dtype: object 

>>> w.multiplier().head(6)
k = 7: k·n has 135 bits, k·n mod 8 = 5 → Q1 polynomials, plain threshold
    k     score  kn mod 8  kn bits  chosen
0   7  6.833740         5      135    True
1  67  6.459246         1      139   False
2  43  6.315669         1      138   False
3   2  6.066391         6      134   False
4   3  5.905973         1      134   False
5  15  5.900268         5      137   False 

>>> fb = w.factor_base(); fb.head(10)
400 primes, largest 6,037; large-prime bound 241,480; primes below 30 are not sieved; plain threshold 54.3 bits; need 433 relations
    p  root  p - root  legendre(kn, p)    log2 p  byte log  sieved
j                                                                 
0   2     1         1                1  1.000000         1   False
1   3     1         2                1  1.584963         2   False
2   5     3         2                1  2.321928         2   False
3   7     0         0                0  2.807355         3   False
4  11     4         7                1  3.459432         3   False
5  13     1        12                1  3.700440         4   False
6  17     1        16                1  4.087463         4   False
7  23     3        20                1  4.523562         5   False
8  31    14        17                1  4.954196         5    True
9  53    15        38                1  5.727920         6    True 

>>> fb.tail(3)
        p  root  p - root  legendre(kn, p)     log2 p  byte log  sieved
j                                                                      
397  6011  2572      3439                1  12.553389        13    True
398  6029  4499      1530                1  12.557703        13    True
399  6037  3722      2315                1  12.559616        13    True 

>>> w.family()
family 1: a = 17476844433867587 (54 bits; target 1.748e+16), s = 5 primes, 16 polynomials; b₀ = 27043964795838022
     j     q  root of kn mod q               B_l  B_l mod q
0   14    83                 3  4000723424620291         80
1  176  2347               703   953147033461888        703
2  230  3307               672  6214928652624216       2635
3  325  4861               171  7974416982990806        171
4  371  5581              1618  7900748702140821       3963 

>>> w.polynomials().head(8)
   t   gray signs of B_l flips B roots move                  b                lin                           c  b² ≡ kn (mod a)
0  0  00000        +++++                     27043964795838022  54087929591676044  -2346106294932794175756795             True
1  1  00001        -++++       0     +delta  19042517946597440  38085035893194880  -2346106316032629648197127             True
2  2  00011        --+++       1     +delta  17136223879673664  34272447759347328  -2346106319978840942413319             True
3  3  00010        +-+++       0     -delta  25137670728914246  50275341457828492  -2346106300624527748120059             True
4  4  00110        +--++       2     +delta  12707813423665814  25415626847331628  -2346106327540941031116219             True
5  5  00111        ---++       0     +delta   4706366574425232   9412733148850464  -2346106335513698138675975             True
6  6  00101        -+-++       1     -delta   6612660641349008  13225321282698016  -2346106334279070542920455             True
7  7  00100        ++-++       0     -delta  14614107490589590  29228214981179180  -2346106324560791157213627             True 

>>> w.roots(0).head(12)
     p  root1  root2  start1  start2  g(root1) mod p  sieved
j                                                           
0    2      1      1       1       1             0.0   False
1    3      0      2       1       0             0.0   False
2    5      3      0       2       4             0.0   False
3    7      4      4       1       1             0.0   False
4   11      6      3       0       8             0.0   False
5   13      4      3       8       7             0.0   False
6   17      3      6      16       2             0.0   False
7   23      2     17      10       2             0.0   False
8   31     27     12      12      28             0.0    True
9   53     39     27      46      34             0.0    True
10  59     35     54      17      36             0.0    True
11  61      9      7      45      43             0.0    True 

>>> s = w.sieve(0); s
polynomial 0: 32,769 positions, 10 reports, 10 survivors
SieveResult(t=0, width=32769, survivors=10, max_sum=58) 

>>> s.acc[:16]  # the first 16 bytes of the sieve array, x = -M ... -M+15
[ 80  80  80  80  90  90  96  95  80  94  80  80 106  88  91  90] 

>>> s.table(10)
     pos      x                        g(x)    log2|g|  sieve_sum  threshold  byte  survives
0   3118 -13266    728869331626255819521273  79.270006         58       48.0   138      True
1   5130 -11254   -133228681202493393557079  76.818251         55       48.0   135      True
2   4469 -11915    134388353418998348015020  76.830754         54       48.0   134      True
3   5846 -10538   -405882416279340951936839  78.425408         52       48.0   132      True
4   7952  -8432  -1103982773245066754259515  79.868992         51       48.0   131      True
5  20175   3791  -2094729614501541575270244  80.793038         50       48.0   130      True
6  24286   7902  -1254396692803996695767559  80.053268         50       48.0   130      True
7   2487 -13897   1028385746643042134582220  79.766656         49       48.0   129      True
8  11355  -5029  -1904374209372030556711404  80.655591         48       48.0   128      True
9  29033  12649    450824055019854459147748  78.576911         48       48.0   128      True 

>>> td = w.trial_divide(0); td.head(12)
polynomial 0: 10 survivors → {'drop': 7, 'two-large': 2, 'one-large': 1}; held: 0 full, 3 partial edges
       x                        g(x)    log2|g|  sieve_sum  threshold                                               base part    cofactor  cofactor bits  \
0 -13897   1028385746643042134582220  79.766656         49       48.0           2^2 · 3^3 · 5 · 79 · 991 · 1021 · 1229 · 5879     3297467             22   
1 -13266    728869331626255819521273  79.270006         58       48.0                3 · 179 · 389 · 463 · 1597 · 1913 · 2081     1185367             21   
2 -11915    134388353418998348015020  76.830754         54       48.0                2^2 · 5 · 211 · 991 · 2939 · 3433 · 4759      669247             20   
3 -11254   -133228681202493393557079  76.818251         55       48.0     -1 · 3 · 13^2 · 179 · 229 · 389 · 397 · 1319 · 3527        8923             14   
4 -10538   -405882416279340951936839  78.425408         52       48.0                -1 · 7 · 179 · 1171 · 1871 · 3331 · 4373    10149961             24   
5  -8432  -1103982773245066754259515  79.868992         51       48.0                 -1 · 5 · 409 · 503 · 1249 · 1801 · 5531    86262331             27   
6  -5029  -1904374209372030556711404  80.655591         48       48.0  -1 · 2^2 · 3 · 7 · 17 · 307 · 347 · 1321 · 2243 · 2693     1568873             21   
7   3791  -2094729614501541575270244  80.793038         50       48.0       -1 · 2^2 · 3 · 7 · 61 · 1021 · 1117 · 3083 · 3889    29897059             25   
8   7902  -1254396692803996695767559  80.053268         50       48.0             -1 · 3 · 101 · 107 · 109 · 367 · 383 · 1747  1445520493             31   
9  12649    450824055019854459147748  78.576911         48       48.0                    2^2 · 101 · 311 · 1021 · 1483 · 5179   457564711             29   

                                                          kind large primes                    y = step·a·x + b mod kn y² ≡ product (mod kn)  
0  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
1  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
2  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
3                                                    one-large         8923  41002535473233943200141433951513856390073                  True  
4  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
5            drop: probable prime, too big for one large prime                                                    None                  None  
6  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
7  drop: cofactor below pmax² but above the bound (it would...                                                    None                  None  
8                                                    two-large  22283 64871                      138129068681217510496                  True  
9                                                    two-large  18979 24109                      221091649208786945985                  True   

>>> td.kind.value_counts()
kind
drop: cofactor below pmax² but above the bound (it would be a prime)    6
two-large                                                               2
one-large                                                               1
drop: probable prime, too big for one large prime                       1
Name: count, dtype: int64 

>>> w.collect()
families sieved      38.00
polynomials         592.00
seconds               0.45
polynomials / s    1319.00
full relations      434.00
needed              433.00
partial edges      1958.00
cycles closed       269.00
workers               1.00
Name: collection, dtype: float64 

>>> w.relation(0)
y                                                                   106182306571349770232
right-hand side               -1 · 3^2 · 5^2 · 59 · 61 · 97 · 269 · 521 · 829 · 1129 ·...
odd columns                   [0, 11, 12, 16, 30, 57, 75, 99, 208, 238, 243, 259, 285,...
large primes (even powers)                                                             ()
y² ≡ rhs (mod kn)                                                                    True
Name: relation 0, dtype: object 

>>> w.relations().head()
   i                                          y  odd columns  nonzero columns  large primes  from partials
0  0                      106182306571349770232           14               16             0          False
1  1  41002535473233943200138425948246961975193           14               14             0          False
2  2  41002535473233943200300610105126206803231           13               15             0          False
3  3  41002535473233943200157072198046062284967           14               15             0          False
4  4  41002535473233943200291093030689460926084           16               17             0          False 

>>> w.partials().head()
   edge  prime u  prime v                                          y
0     0        1     8923  41002535473233943200141433951513856390073
1     1    22283    64871                      138129068681217510496
2     2    18979    24109                      221091649208786945985
3     3        1    10111  41002535473233943200073967008016242387868
4     4        1    10007  41002535473233943200155251621069038953960 

>>> w.cycle_summary()
partial edges                       1958
vertices (large primes, incl. 1)    1764
cycles closed → full relations       269
Name: large-prime graph, dtype: int64 

>>> m = w.matrix(dense=False); m.head()
434 relations × 401 columns; 423 rows after singleton pruning
   relation                                                  odd columns  weight
0         0  [0, 11, 12, 16, 30, 57, 75, 99, 208, 238, 243, 259, 285,...      14
1         1  [0, 6, 7, 12, 15, 46, 137, 168, 174, 238, 259, 285, 371,...      14
2         2  [0, 5, 12, 124, 207, 215, 225, 238, 259, 285, 286, 363, ...      13
3         3  [0, 6, 12, 14, 40, 48, 117, 238, 239, 259, 285, 320, 333...      14
4         4  [0, 2, 4, 8, 12, 18, 47, 51, 87, 184, 238, 259, 284, 285...      16 

>>> w.dependencies().head()
36 dependencies
   dependency  relations                                                         rows
0           0        198  [0, 4, 6, 8, 9, 10, 12, 13, 15, 16, 20, 22, 23, 24, 25, ...
1           1        196  [0, 2, 3, 4, 5, 8, 9, 14, 15, 17, 19, 20, 23, 24, 26, 27...
2           2        176  [2, 5, 6, 10, 12, 13, 16, 19, 21, 25, 26, 27, 30, 34, 35...
3           3        178  [1, 2, 4, 6, 8, 11, 12, 15, 16, 17, 18, 20, 23, 24, 27, ...
4           4        202  [0, 1, 4, 5, 7, 8, 11, 12, 13, 15, 16, 18, 19, 20, 22, 2... 

>>> w.square_root(0)
relations                                                             198
all_even                                                             True
X                                1810505355281038925296728896486864313330
Y                                3279719630893588208427609233365508102283
X2_eq_Y2_mod_kn                                                      True
gcd(X-Y, n)                                          94866163883639224007
gcd(X+Y, n)                                          61744934419289236901
factor                                               94866163883639224007
X mod n == ±Y mod n (trivial)                                       False
Name: dependency 0, dtype: object 

>>> w.factors()
dependency 0 splits n: 94866163883639224007 × 61744934419289236901
[94866163883639224007, 61744934419289236901] 

