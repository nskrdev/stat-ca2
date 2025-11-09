import json

# Read the existing notebook
with open('CA2_Statistical_Analysis.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cells for ANOVA visualization and assumptions
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.2 ANOVA Visualization\n",
            "\n",
            "Visuali 0.05:\n >vene_p   "if le,
         \")\n"ion:\\n Conclustatisticalprint(\"S    "
          "\n",         ")\n",
 \\n \"\\"*80 +\"= " +\"\\nt(\"prin     
        "\n",       ",
    )\n05\"): 0.vel (αlence Significat(f\"     "prin     )\n",
  _p:.6f}\"ene: {levvalue\"P-(f     "print
       ,\")\n":.4f}vene_stattistic: {lestavene's int(f\"Le       "pr   ,
  )\n"\"\"\\n"=\"*80 + print(\   "       \n",
  \")ene's Test)t (LevTesce  of Varianomogeneity"print(\"H           "\n",
          ,
   nent)\n"_by_contideath_rates= levene(*p  levene_vene_stat, "le        e\n",
   arianc of v homogeneity forevene's test Perform L   "#       e": [
  "sourc      
  s": [],    "output   
 a": {},"metadat        ,
": Nonetcution_coun    "exe,
    de""coype":  "cell_t  {
        },
  
           ]"
nts.ontineacross cqual re eeath rates aces of drianst if the va te"We         "\n",
           \n",
    vene's Test)iance (Leof Vareneity 2: Homogon mpti"### Assu            ource": [
  "s},
      ta": {ada"met
        ",kdownmartype": "     "cell_
    {
    ]
    },  "
     \")le sizes.rger samplah ly witiality, espec normalons of to violatiusttively rob is relaNote: ANOVA"\\n  "print(\     ",
     \")\nnistributionormal dfrom eviates ficantly d: Data signi 0.05p-value <"- If (\    "print,
        \")\n"stributionl di with norma consistentata ise > 0.05: D If p-valu"-int(\ "pr         ",
  \")\nrpretation:nte(\"Iint  "pr         ")\n",
 \\n\ + \"\"*80=\" + \"\"\\n "print(
           ",\n "        n",
   ")\':<15}\pleso few sam{'To15} 'N/A':<':<15} {20} {'N/At:<inen\"{contt(f  prin         "  ",
       \ne:   "    els
         \")\n",<15} {is_normal:val:<15.6f}{p_at:<15.4f} t:<20} {w_st{continen\"t(f       prin       " 
     rmal}\n",s_noal': i'normval, p_: at, 'p' {'w': w_st] =ontinent[cesultsality_rrm"        no       
     No\"\n",\".05 else l > 0f p_va" iYes\"normal = \   is_              "",
   \nath_rates)apiro(del = sh, p_vaw_stat          "     \n",
     es) >= 3:ath_raten(deif l        " 
       = 3)\n",mples (n >ough saf we have enrform test iOnly pe   "    #    
       \n",  "            \n",
 ues].valp'ths/1M poinent]['Deat'] == conttinen['Conova[df_ans = df_anovaeath_rate"    d        \n",
    ue()):nent'].uniqanova['Conti sorted(df_inent inont c"for          
  {}\n",lts = ty_resumalior"n    ",
        n"\      ",
      " * 80)\n\"-\"print(            
",")\n':<15}\ormal?} {'N15value':<'P- {15}tic':< {'W-statis:<20}tinent'"{'Conf\"print(         )\n",
   \\n\""0 + \"=\"*8int(\        "prn",
    nt\")\ntine) by ColkWiapiro-t (Shrmality Tes"No\"print(           ": [
 rce "sou   ,
    []utputs": "o       },
 : {ata" "metad       None,
 _count":ecution      "ex",
  e": "codecell_typ" {
        
    ]
    },       ibution."
l distrrmanollow a nt foch contineeahin ates with ratf de ie test  "W
              "\n",       )\n",
 o-Wilkst (Shapirity Te1: Normaltion sump### As       "
     ource": [       "s": {},
 tadata "me,
       "own": "markdpe_ty  "cell
      
    {},     ]
    "
   ns. assumptioeck these ch    "Let's
        \n",  "         ual\n",
 oximately eqpprd be aps shoulougrs crosnces a: Variance** of VariaHomogeneity **     "2.,
       uted\n"stribally dinormy roximateld be appulsho group ithin eachta wity**: DaalrmNo "1. **         
  s:\n",mptionn assutwo maiA requires    "ANOV,
          "\n"          ck\n",
 mptions Che3 ANOVA Assu 3.##      " [
      ":ceur        "so
data": {}, "meta",
       : "markdown""cell_type           {
 
    },

        ]")" outliers\e potentialarskers  beyond whioints\"- Print(  "p     ,
     ")\n"s\ilequartthe QR from  Id to 1.5 × extenhiskerst(\"- Wprin "       \n",
    nent\")ontich cmean for ea show the kersdiamond mare red \"- Th   "print(      ,
   \n"")ian\ medbox is theinside the - The line rint(\"      "p    \n",
  ")le\rcentio 75th peQR): 25th trange (Irquartile  the inteox shows(\"- The bint   "pr      
   \n",tion:\") InterpretaPlotnBox \\print(\" "    ",
             "\n    ,
  "w()\n   "plt.sho     \n",
    _layout()"plt.tight           
 ",ht')\n='upper rigd(locegen   "plt.l  
       "\n",      
      n",)\=1.5hswidtlack', line='blorsn', edgecobel='Mea', la='D, markerorder=3 zs=100,r='red', olomeans, cs, (positiontterlt.sca  "p
          n",)\ans)(len(me= range"positions      
       \n",'].mean()M popeaths/1')['Dinentontpby('Ca.groudf_anov= eans   "m         s\n",
 an marker me     "# Add     
  ",      "\n",
      )\nyle='--' linest3,a=0., alphis='y'grid(ax      "plt.    n",
  'right')\ion=45, ha=s(rotatt.xtick"pl           
 \n",ld')ght='bo=13, fontweize', fontsionpulatiillion Po per M'Deathsplt.ylabel(   "
         ld')\n",ght='bontwei=13, fofontsizeontinent', el('C "plt.xlab      ",
     \n pad=20)ight='bold',16, fontwee=ntsiz, fontinents'Cooss ates Acreath Rf Dstribution otitle('Dilt.        "p",
    \nmize plot"# Custo          \n",
       "",
       e='Set2')\n, palettaths/1M pop'ent', y='Dein x='Cont=df_anova,ot(datasns.boxpl   "     \n",
    lotox p"# Create b      ,
          "\n"",
        , 7))\n(12gsize=e(fifigur    "plt.",
        nts\ncontiness acros ath rateing deot compar box plreate    "# C     
   ce": ["sour
        ],tputs": [      "oua": {},
     "metadat
     nt": None,on_cou"executi        
"code",": ll_type    "ce {
    
   },         ]
 plot."
  sing a box continents uross h rates acn of deatistributioze the d