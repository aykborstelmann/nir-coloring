\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ivs-thesis}[2023/06/19 IVS Thesis]

\DeclareOption{logo}{\gdef\@uselogo{yes}}
\DeclareOption{nologo}{\global\let\@uselogo\@empty}

\DeclareOption{ba}{\gdef\@thesistype{\GetTranslation{ba-thesis}}}
\DeclareOption{ma}{\gdef\@thesistype{\GetTranslation{ma-thesis}}}

\ExecuteOptions{ma,logo} 

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions \relax 


\LoadClass[twoside,11pt,a4paper]{book}

\RequirePackage{translations} 

\DeclareTranslation{English}{ba-thesis}{Bachelor's Thesis} 
\DeclareTranslation{German}{ba-thesis}{Bachelorarbeit} 
\DeclareTranslation{English}{ma-thesis}{Master's Thesis} 
\DeclareTranslation{German}{ma-thesis}{Masterarbeit} 
\DeclareTranslation{English}{written-submitted}{written and submitted by} 
\DeclareTranslation{German}{written-submitted}{vorgelegt von} 
\DeclareTranslation{English}{examiner}{Examiner}
\DeclareTranslation{German}{examiner}{Pr\"{u}fer/-in}
\DeclareTranslation{English}{supervisors}{Supervisor(s)} 
\DeclareTranslation{German}{supervisors}{Betreuer/-innen} 
\DeclareTranslation{English}{born}{born}
\DeclareTranslation{German}{born}{geboren am} 
\DeclareTranslation{English}{abstractname}{Abstract}
\DeclareTranslation{German}{abstractname}{Zusammenfassung}
\DeclareTranslation{English}{matriculationnumber}{Matriculation number}
\DeclareTranslation{German}{matriculationnumber}{Matrikelnummer}

\RequirePackage{fmtcount}

\newcommand{\setthesistype}[1]{\def\@thesistype{#1}}
\let\thesistype\setthesistype
\newcommand{\show@thesistype}{\@thesistype}

%\newcommand{\uselogo}{\def\@uselogo{yes}}

\global\let\@examiner\@empty 
\newcommand{\setexaminer}[1]{\def\@examiner{#1}}
\let\examiner\setexaminer

\global\let\@supervisor\@empty
\newcommand{\setsupervisor}[1]{\def\@supervisor{#1}}
\let\supervisor\setsupervisor

\newcommand{\birthinfo}[1]{\normalsize \GetTranslation{born} #1}
\newcommand{\matricnumber}[1]{\normalsize \GetTranslation{matriculationnumber}: #1}

\let\oldmaketitle\maketitle
\renewcommand{\maketitle}{
	\begin{titlepage}
		\ifx\@uselogo\@empty
			\null\vskip16mm
		\else 
			\begin{flushright}
				\includegraphics[height=16mm]{gfx/ubn-logo}
				\includegraphics[height=16mm]{gfx/IVS-4-100}\par
			\end{flushright}
		\fi 
		\null\vfil\vskip60pt
		\centering 
		\begin{minipage}{0.618034\paperwidth}
			\centering
			{\Large \textsc{\show@thesistype}\par}\vspace{1ex}
			\rule{\linewidth}{1pt}\\
			{\Huge\bfseries\@title\par}
			\rule{\linewidth}{1pt}\vspace{2cm}
		\end{minipage}\par
	
		{\emph{\GetTranslation{written-submitted}}\\\vspace{1em}
		\Large 
		\begin{tabular}[t]{c}
			\@author
		\end{tabular}\par}\vspace{2cm}
	
		{\def\and{\end{tabular}\\ \personprefix & \renewcommand{\arraystretch}{1.0}\begin{tabular}[t]{l} }
		\def\arraystretch{1.5}
		\newcounter{examinercnt} 
		\setcounter{examinercnt}{0}
		\gdef\personprefix{\stepcounter{examinercnt}\ordinal{examinercnt} \GetTranslation{examiner}:}
		\Large 
		\global\let\@needtable\@empty
		\ifx\@examiner\@empty\else
			\gdef\@needtable{yes}
		\fi 
		\ifx\@supervisor\@empty\else 
			\gdef\@needtable{yes}
		\fi 
		\ifx\@needtable\@empty\else%
			\begin{tabular}[t]{rl}
				\ifx\@examiner\@empty\else\personprefix & \renewcommand{\arraystretch}{1.0}\begin{tabular}[t]{l}\@examiner\end{tabular}\fi\renewcommand{\arraystretch}{1.5}\\
				\ifx\@supervisor\@empty\else 
					\gdef\personprefix{}
	%				\makeatletter
	%				\global\def\personprefix\@empty 
	%				\makeatother
					\GetTranslation{supervisors}: & \renewcommand{\arraystretch}{1.0}\begin{tabular}[t]{l}\@supervisor\end{tabular}\renewcommand{\arraystretch}{1.5}
				\fi
			\end{tabular}%
		\fi}
		\vfill\null
		{\@date\par}
	\end{titlepage}
}

\RequirePackage{iftex} 

\RequirePackage[margin=3cm,bindingoffset=3mm,includehead,includefoot]{geometry}

\RequirePackage{fancyhdr,emptypage,xcolor,graphicx,subcaption,booktabs}
\RequirePackage[labelfont=bf,format=hang]{caption}

\definecolor{ub-blue}{cmyk}{1,0.7,0,0}
\definecolor{ub-yellow}{cmyk}{0,0.3,1,0}
\definecolor{ub-gray}{cmyk}{0,0,0.15,0.55}

\RequirePackage[hidelinks,breaklinks]{hyperref}
\hypersetup{colorlinks=true,
            linkcolor=black,
            filecolor=black,
            urlcolor=ub-blue,
            citecolor=ub-blue, %black, %darkblue,
            bookmarksnumbered=true
            }


\fancypagestyle{ivs-fancy}{
	\fancyhf{}
	\fancyfoot[LE,RO]{\sffamily\small\thepage}
	\fancyhead[LE]{\sffamily\small\leftmark}
	\fancyhead[RO]{\sffamily\small\rightmark} 
	\renewcommand{\headrulewidth}{0pt} 
	\renewcommand{\footrulewidth}{0pt} 
}

\fancypagestyle{plain}{
	\fancyhf{}
	\fancyfoot[LE,RO]{\sffamily\small\thepage}
	\renewcommand{\headrulewidth}{0pt} 
	\renewcommand{\footrulewidth}{0pt} 
}

\setlength{\headheight}{13pt} 

\pagestyle{ivs-fancy} 
\AtBeginDocument{\thispagestyle{plain}} 

\RequirePackage[intlimits]{mathtools}

\RequirePackage[amsmath,thmmarks]{ntheorem}

\newenvironment{abstract}{
\small%
\begin{center}
	\bfseries\GetTranslation{abstractname}
\end{center}%
\quotation%
}{\endquotation} 

\ifPDFTeX
	\wlog{PDF LaTeX detected}
	\RequirePackage[utf8]{inputenc}
	\RequirePackage[T1]{fontenc}
	\RequirePackage{textcomp,tgpagella} 
\else 
	\ifLuaTeX
		\wlog{LuaLaTeX detected}
		\RequirePackage{fontspec}
		\setmainfont[Ligatures=TeX]{TeX Gyre Pagella} 
	\else 
		\ifXeTeX
			\wlog{XeLaTeX detected}
			\RequirePackage{fontspec}
			\setmainfont[Ligatures=TeX]{TeX Gyre Pagella} 
		\fi 
	\fi 
\fi 
