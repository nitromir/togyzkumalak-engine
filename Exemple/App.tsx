import { useState, useEffect } from 'react';
import { companies } from './data';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Users, Wallet, TrendingUp, ChevronDown, ChevronUp, Sparkles, Zap, AlertTriangle, MessageSquare, Loader2, Terminal, Sun, Moon, FileCode2, UserCheck, Mic, BookOpen, Settings2, Star, Target, FileText, Save, Copy, RefreshCw, Building2, HelpCircle } from 'lucide-react';
import { generateCoverLetter, generateTrojanHorseStrategy, generatePersonaIntro, generateInterviewPhrases, generatePersonaAnalysis, updateModel, getModelName, generateTailoredResume, generateMarketAnalysis } from '@/lib/gemini';
import { cn } from "@/lib/utils";
import { CompanyManagerTab } from '@/components/CompanyManagerTab';
import { SystemHelpDialog } from '@/components/SystemHelpDialog';
import { AIContentViewer } from '@/components/AIContentViewer';
import { UserProfileEditor } from '@/components/UserProfileEditor';
import { ConfigViewerDialog } from '@/components/ConfigViewerDialog';
import { LinkedInPostGenerator } from '@/components/LinkedInPostGenerator';
import { PolyglotModal } from '@/components/PolyglotModal';
import { useLessonStore } from '@/stores/useLessonStore';
// LoginForm removed
import { API_URL } from '@/config';

// Retro Pixel Icons
const RETRO_ICONS = ["👾", "🕹️", "💾", "📼", "📟", "🤖", "🚀", "🛸", "👻", "💀", "👽", "👿", "🤡", "👹"];

const getRandomRetroIcon = (seed: string) => {
    const index = seed.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % RETRO_ICONS.length;
    return RETRO_ICONS[index];
}

const CustomTooltip = ({ active, payload, theme }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className={cn(
        "p-3 border rounded shadow-2xl text-sm font-mono z-50 relative",
        theme === 'dark' 
          ? "bg-zinc-950 text-white border-purple-900/50" 
          : "bg-white text-slate-900 border-slate-200"
      )}>
        <p className={cn("font-bold mb-1", theme === 'dark' ? "text-green-400" : "text-purple-600")}>{data.name}</p>
        <p className="flex justify-between gap-4"><span>PROB:</span> <span className="font-bold">{data.metrics.hiring_probability}%</span></p>
        <p className="flex justify-between gap-4"><span>SALARY:</span> <span className={cn("font-bold", theme === 'dark' ? "text-purple-400" : "text-emerald-600")}>${data.metrics.salary_estimated_usd.toLocaleString()}</span></p>
      </div>
    );
  }
  return null;
};

const RisingLinesBackground = () => {
    return (
        <div className="absolute inset-0 z-0 rising-lines-container px-4 pointer-events-none">
            {[...Array(20)].map((_, i) => (
                <div 
                    key={i} 
                    className="rising-line" 
                    style={{ 
                        animationDelay: `${Math.random() * 2}s`,
                        height: `${Math.random() * 50 + 20}%`
                    }} 
                />
            ))}
        </div>
    );
};

function MarketAnalysisTab({ theme }: { theme: string }) {
    const [loading, setLoading] = useState(false);
    const [analysisData, setAnalysisData] = useState<{chartData: any[], analysis: string} | null>(null);

    const handleAnalyze = async () => {
        setLoading(true);
        // Flatten all roles from all companies
        const allJobs = companies.flatMap(c => c.open_roles || []);
        const result = await generateMarketAnalysis(allJobs);
        setAnalysisData(result);
        setLoading(false);
    };

    return (
        <div className="space-y-6">
            <Card className="border-border bg-card/50 backdrop-blur-md">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 uppercase tracking-widest font-bold text-sm text-cyan-500">
                        <TrendingUp className="h-4 w-4" /> Market_Pulse_Analysis
                    </CardTitle>
                    <CardDescription className="font-mono text-xs">
                        Analyze skill demand across current database vacancies.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {!analysisData ? (
                        <div className="text-center py-12">
                            <Button 
                                onClick={handleAnalyze} 
                                disabled={loading}
                                className="bg-cyan-600 hover:bg-cyan-500 text-white font-bold font-mono uppercase tracking-widest h-12 px-8 rounded shadow-[0_0_20px_rgba(8,145,178,0.4)] relative overflow-hidden"
                            >
                                {loading ? (
                                    <>
                                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 via-blue-500 to-cyan-600 animate-pulse" />
                                        <span className="relative z-10 flex items-center justify-center">
                                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                                            Scanning_Market_Data...
                                        </span>
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="mr-2 h-5 w-5" />
                                        Initialize_Market_Scan
                                    </>
                                )}
                            </Button>
                            
                            {loading && (
                                <div className="mt-4 space-y-3">
                                    <div className="flex items-center justify-center gap-2">
                                        <div className="h-2 w-2 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                        <div className="h-2 w-2 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                        <div className="h-2 w-2 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                                    </div>
                                    <p className="text-xs font-mono text-cyan-500 animate-pulse">
                                        🔍 Analyzing {companies.reduce((acc, c) => acc + (c.open_roles?.length || 0), 0)} job requirements...
                                    </p>
                                    <p className="text-[10px] font-mono text-muted-foreground">
                                        AI is extracting skill patterns...
                                    </p>
                                </div>
                            )}
                            
                            {!loading && (
                                <p className="mt-4 text-xs font-mono text-muted-foreground">
                                    Scanning {companies.reduce((acc, c) => acc + (c.open_roles?.length || 0), 0)} active role vectors...
                                </p>
                            )}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 animate-in fade-in duration-500">
                            {/* Chart Section */}
                            <div className="h-[300px] relative">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={analysisData.chartData}>
                                        <PolarGrid stroke={theme === 'dark' ? "#3f3f46" : "#cbd5e1"} />
                                        <PolarAngleAxis 
                                            dataKey="subject" 
                                            tick={{ fill: theme === 'dark' ? "#a1a1aa" : "#475569", fontSize: 10, fontFamily: 'monospace' }} 
                                        />
                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                        <Radar
                                            name="Market Demand"
                                            dataKey="A"
                                            stroke="#06b6d4"
                                            strokeWidth={2}
                                            fill="#06b6d4"
                                            fillOpacity={0.3}
                                        />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: theme === 'dark' ? '#18181b' : '#fff', borderColor: '#3f3f46', fontFamily: 'monospace', fontSize: '12px' }}
                                            itemStyle={{ color: theme === 'dark' ? '#22d3ee' : '#0891b2' }}
                                        />
                                    </RadarChart>
                                </ResponsiveContainer>
                                <div className="absolute top-0 right-0">
                                    <Button variant="ghost" size="sm" onClick={handleAnalyze} className="text-[10px] uppercase">
                                        <RefreshCw className="w-3 h-3 mr-1" /> Rescan
                                    </Button>
                                </div>
                            </div>

                            {/* Analysis Text with Expand */}
                            <div className="flex flex-col h-full">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-4 flex items-center gap-2">
                                    <Terminal className="w-3 h-3" /> Strategic_Insight
                                </h4>
                                <AIContentViewer 
                                    content={analysisData.analysis} 
                                    title="Market Pulse - Strategic Analysis"
                                />
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}

export default function App() {
  // Authentication DISABLED
  const [isAuthenticated] = useState(true);
  
  const [isGeneratorOpen, setIsGeneratorOpen] = useState(false);
  const [isTrojanOpen, setIsTrojanOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [currentModel, setCurrentModel] = useState(getModelName());
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<{count: number, message: string} | null>(null);
  const [userProfile, setUserProfile] = useState<any>(null);
  const [selectedTargets, setSelectedTargets] = useState<string[]>(() => {
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem('selectedTargets');
        return saved ? JSON.parse(saved) : [];
    }
    return [];
  });
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('theme') || 'dark';
    }
    return 'dark';
  });

  // Login handler removed

  // All hooks must be called before any conditional returns
  useEffect(() => {
    localStorage.setItem('selectedTargets', JSON.stringify(selectedTargets));
  }, [selectedTargets]);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Load user profile on mount
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const response = await fetch(`${API_URL}/api/user-profile`);
        if (response.ok) {
          const data = await response.json();
          console.log('✅ App.tsx: Profile loaded from backend:', data);
          setUserProfile(data);
        } else {
          throw new Error('Backend not available');
        }
      } catch (error) {
        console.warn('⚠️ App.tsx: Backend unavailable, trying localStorage');
        // Fallback to localStorage
        const saved = localStorage.getItem('user_profile');
        if (saved) {
          const profileData = JSON.parse(saved);
          console.log('✅ App.tsx: Profile loaded from localStorage:', profileData);
          setUserProfile(profileData);
        } else {
          console.warn('⚠️ App.tsx: No profile in localStorage, using defaults');
          // Set default profile
          const defaultProfile = {
            core_ai_skills: ["LangChain", "LLM", "TTS/STT", "Voice AI", "Realtime Assistants"],
            technical_skills: ["Python", "React", "React Native", "Vibecoding"],
            business_skills: ["Agency Management", "SEO/SERM", "PR/Marketing", "Programmatic SEO"],
            target_role: "AI Product Manager",
            years_experience: 15,
            unique_value_prop: "Bridge between AI technical implementation and business growth"
          };
          setUserProfile(defaultProfile);
        }
      }
    };
    loadProfile();
  }, []);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleModelChange = (model: string) => {
      updateModel(model);
      setCurrentModel(model);
  }

  const toggleTarget = (companyName: string) => {
      setSelectedTargets(prev => 
          prev.includes(companyName) 
            ? prev.filter(c => c !== companyName)
            : [...prev, companyName]
      );
  }

  const idealPersona = {
      role: "AI Product Manager",
      top_skills: ["LangChain/RAG", "Python (Vibecoding)", "Agency/Client Comms", "Generative AI"],
      mindset: "Builder & Shipper (0->1)",
      common_pain_points_solved: ["Integration friction", "User Trust/Hallucination", "Enterprise Security"]
  };

  const userBackground = {
      role: "Agency Owner & AI Builder",
      skills: ["SEO", "Marketing", "No-Code/Low-Code", "Voice AI", "Team Management"],
      unique_value: "Can build prototypes AND sell them."
  };

  return (
    <div className="min-h-screen flex flex-col font-mono p-4 md:p-8 selection:bg-purple-900 selection:text-white transition-colors duration-300 bg-arcade-pattern">
      <header className="shrink-0 mb-6 md:mb-10 border-b border-border/60 pb-4 md:pb-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 backdrop-blur-sm bg-background/80 p-4 -mx-4 md:mx-0 md:p-0 rounded-b-xl md:rounded-none">
      <div>
            <h1 className="text-3xl md:text-5xl font-black tracking-tighter mb-2 flex items-center gap-2 md:gap-3">
                <Terminal className="h-8 w-8 md:h-10 md:w-10 text-primary animate-pulse" />
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-500">AI_PM_PATHFINDER</span> 
                <span className="hidden md:inline-block text-xs md:text-sm font-mono font-normal text-muted-foreground border border-border px-2 py-1 rounded">v1.0.8_RETRO_CORE</span>
            </h1>
            <p className="text-muted-foreground text-sm md:text-lg tracking-wide pl-0 md:pl-14">
                SYSTEM STATUS: <span className="text-primary font-bold">ONLINE</span> // AI_MODEL: <span className="text-purple-500">{currentModel}</span>
        </p>
      </div>
        <div className="flex gap-2 shrink-0 self-end md:self-auto">
            <ConfigViewerDialog theme={theme} />
            <SystemHelpDialog theme={theme} />
            
            <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
                <DialogTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full w-10 h-10 border-2 bg-background/50">
                        <Settings2 className="h-5 w-5 text-muted-foreground" />
                    </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px] font-mono">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2"><Settings2 className="w-4 h-4"/> AI_CORE_CONFIG</DialogTitle>
                        <DialogDescription>Select the LLM model for generation.</DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <div className="space-y-2">
                            <label className="text-xs font-bold uppercase text-muted-foreground">Active Model</label>
                            <select 
                                className="w-full p-2 border border-input bg-background rounded text-sm focus:ring-1 focus:ring-primary outline-none"
                                value={currentModel}
                                onChange={(e) => handleModelChange(e.target.value)}
                            >
                                <option value="gemini-2.0-flash">gemini-2.0-flash (Recommended)</option>
                                <option value="gemini-1.5-flash">gemini-1.5-flash (Fast)</option>
                                <option value="gemini-1.5-pro">gemini-1.5-pro (Reasoning)</option>
                                <option value="gemini-3-pro-preview">gemini-3-pro-preview (Experimental)</option>
                                <option value="gemini-2.5-flash-native-audio-preview-09-2025">gemini-2.5-flash-audio (Voice Sim)</option>
                            </select>
                            <p className="text-[10px] text-muted-foreground">
                                Note: "gemini-3-pro-preview" might require special access or fallback to stable models if unavailable.
                            </p>
                        </div>
                    </div>
                </DialogContent>
            </Dialog>
            <Button variant="outline" size="icon" onClick={toggleTheme} className="rounded-full w-10 h-10 border-2 bg-background/50">
            {theme === 'dark' ? <Sun className="h-5 w-5 text-yellow-400" /> : <Moon className="h-5 w-5 text-purple-600" />}
        </Button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8 relative z-10">
        {/* Left Column: Main Content */}
        <div className="lg:col-span-2">
          
          <Tabs defaultValue="dashboard" className="w-full flex flex-col">
            <TabsList className="shrink-0 grid w-full grid-cols-3 lg:grid-cols-7 bg-muted/80 border border-border p-1 h-auto mb-6 backdrop-blur-sm">
                <TabsTrigger value="dashboard" className="text-[10px] md:text-xs uppercase tracking-widest py-2">Dashboard</TabsTrigger>
                <TabsTrigger value="targets" className="text-[10px] md:text-xs uppercase tracking-widest py-2">Target Lists</TabsTrigger>
                <TabsTrigger value="company-manager" className="text-[10px] md:text-xs uppercase tracking-widest py-2 flex gap-2 items-center justify-center"><Building2 className="w-3 h-3"/> Companies</TabsTrigger>
                <TabsTrigger value="market-pulse" className="text-[10px] md:text-xs uppercase tracking-widest py-2 flex gap-2 items-center justify-center"><TrendingUp className="w-3 h-3"/> Market Pulse</TabsTrigger>
                <TabsTrigger value="persona" className="text-[10px] md:text-xs uppercase tracking-widest py-2 flex gap-2 items-center justify-center"><UserCheck className="w-3 h-3"/> Persona Opt.</TabsTrigger>
                <TabsTrigger value="resume-ops" className="text-[10px] md:text-xs uppercase tracking-widest py-2 flex gap-2 items-center justify-center"><FileText className="w-3 h-3"/> Resume Ops</TabsTrigger>
                <TabsTrigger value="voice-sim" className="text-[10px] md:text-xs uppercase tracking-widest py-2 flex gap-2 items-center justify-center"><Mic className="w-3 h-3"/> Voice Sim</TabsTrigger>
            </TabsList>

            <TabsContent value="dashboard" className="pr-2 space-y-6 data-[state=inactive]:hidden">
          {/* Scatter Plot */}
                <Card className="shadow-2xl backdrop-blur-md bg-card/70 border-border relative overflow-hidden">
                    <RisingLinesBackground />
                    <CardHeader className="border-b border-border/50 p-4 md:p-6 relative z-10">
              <CardTitle className="flex items-center gap-2 text-primary font-mono uppercase tracking-widest text-xs md:text-sm">
                <TrendingUp className="h-4 w-4" />
                Target_Acquisition_Matrix
              </CardTitle>
              <CardDescription className="font-mono text-[10px] md:text-xs">
                X: PROBABILITY_FIT | Y: COMP_PACKAGE_USD
              </CardDescription>
                        
                        {/* Stats Grid */}
                        <div className="grid grid-cols-4 gap-2 md:gap-3 mt-4">
                            <div className="bg-cyan-500/5 border border-cyan-500/20 rounded p-2 text-center">
                                <div className="text-xl md:text-3xl font-bold text-cyan-500 font-mono">{companies.length}</div>
                                <div className="text-[8px] md:text-[10px] text-muted-foreground uppercase tracking-wider font-bold mt-1">Total Companies</div>
                            </div>
                            <div className="bg-yellow-500/5 border border-yellow-500/20 rounded p-2 text-center">
                                <div className="text-xl md:text-3xl font-bold text-yellow-500 font-mono">{selectedTargets.length}</div>
                                <div className="text-[8px] md:text-[10px] text-muted-foreground uppercase tracking-wider font-bold mt-1">Priority Targets</div>
                            </div>
                            <div className="bg-green-500/5 border border-green-500/20 rounded p-2 text-center">
                                <div className="text-xl md:text-3xl font-bold text-green-500 font-mono">
                                    {companies.reduce((sum, c) => sum + (c.open_roles?.length || 0), 0)}
                                </div>
                                <div className="text-[8px] md:text-[10px] text-muted-foreground uppercase tracking-wider font-bold mt-1">Open Roles</div>
                            </div>
                            <div className="bg-purple-500/5 border border-purple-500/20 rounded p-2 text-center">
                                <div className="text-xl md:text-3xl font-bold text-purple-500 font-mono">
                                    {Math.round(companies.reduce((sum, c) => sum + c.metrics.hiring_probability, 0) / companies.length)}%
                                </div>
                                <div className="text-[8px] md:text-[10px] text-muted-foreground uppercase tracking-wider font-bold mt-1">Avg Match</div>
                            </div>
                        </div>
            </CardHeader>
                    <CardContent className="h-[400px] p-6">
                        <div className="w-full h-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#27272a' : '#e2e8f0'} opacity={0.5} />
                  <XAxis 
                    type="number" 
                    dataKey="metrics.hiring_probability" 
                    name="Probability" 
                    unit="%" 
                    domain={[0, 100]} 
                    stroke={theme === 'dark' ? '#71717a' : '#64748b'}
                    tick={{fill: theme === 'dark' ? '#71717a' : '#64748b', fontSize: 12, fontFamily: 'monospace'}}
                    label={{ value: 'HIRING_PROBABILITY', position: 'insideBottomRight', offset: -5, fill: theme === 'dark' ? '#52525b' : '#94a3b8', fontSize: 10 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="metrics.salary_estimated_usd" 
                    name="Salary" 
                    unit="$" 
                    domain={['auto', 'auto']}
                    stroke={theme === 'dark' ? '#71717a' : '#64748b'}
                    tick={{fill: theme === 'dark' ? '#71717a' : '#64748b', fontSize: 12, fontFamily: 'monospace'}}
                    label={{ value: 'EST_SALARY_USD', angle: -90, position: 'insideLeft', fill: theme === 'dark' ? '#52525b' : '#94a3b8', fontSize: 10 }} 
                  />
                  <Tooltip content={<CustomTooltip theme={theme} />} cursor={{ strokeDasharray: '3 3', stroke: theme === 'dark' ? '#10b981' : '#9333ea' }} />
                            <Scatter name="Companies" data={companies} isAnimationActive={true} animationDuration={1500} animationEasing="ease-out">
                    {companies.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={
                            theme === 'dark' 
                                            ? (entry.segment === 'Global CIS' ? '#22c55e' : '#a855f7') 
                                            : (entry.segment === 'Global CIS' ? '#16a34a' : '#d97706')
                        } 
                                    strokeWidth={selectedTargets.includes(entry.name) ? 3 : 1}
                                    stroke={selectedTargets.includes(entry.name) ? "#eab308" : undefined}
                                    className="hover:opacity-80 transition-all duration-300 cursor-pointer animate-in zoom-in duration-500"
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
                        </div>
            </CardContent>
          </Card>

                {/* Priority Targets List */}
                {/* User Profile Editor */}
                <UserProfileEditor 
                    theme={theme} 
                    onSave={(profile) => {
                        console.log('Profile updated:', profile);
                        setUserProfile(profile); // Update App state
                    }}
                />

                {selectedTargets.length > 0 && (
                    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="flex items-center gap-2 mb-4 p-2 bg-muted/20 border-b border-border">
                            <Target className="w-4 h-4 text-yellow-500" />
                            <h3 className="font-mono uppercase tracking-widest text-xs font-bold text-muted-foreground">Priority Targets Protocol</h3>
                        </div>
                        <div className="grid gap-4">
                            {companies.filter(c => selectedTargets.includes(c.name)).map(company => (
                                <CompanyCard 
                                    key={company.name} 
                                    company={company} 
                                    variant={company.segment === 'Global CIS' ? 'cis' : 'enterprise'} 
                                    theme={theme}
                                    isSelected={true}
                                    onToggle={toggleTarget}
                                />
                            ))}
                        </div>
                    </div>
                )}
            </TabsContent>

            <TabsContent value="targets" className="data-[state=inactive]:hidden">
                <Tabs defaultValue="list-a" className="flex flex-col">
                    <TabsList className="grid w-full grid-cols-2 mb-4">
                        <TabsTrigger value="list-a" className="text-xs uppercase tracking-widest font-bold data-[state=active]:bg-emerald-500 data-[state=active]:text-white">
                            <span className="flex items-center gap-2">
                                <Badge variant="outline" className="bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/50">LIST A</Badge>
                                High Probability (CIS)
                            </span>
              </TabsTrigger>
                        <TabsTrigger value="list-b" className="text-xs uppercase tracking-widest font-bold data-[state=active]:bg-purple-500 data-[state=active]:text-white">
                            <span className="flex items-center gap-2">
                                <Badge variant="outline" className="bg-purple-500/20 text-purple-600 dark:text-purple-400 border-purple-500/50">LIST B</Badge>
                                High Salary (Enterprise)
                            </span>
              </TabsTrigger>
            </TabsList>

                    <TabsContent value="list-a" className="pr-2 space-y-4 data-[state=inactive]:hidden">
                        <div className="mb-4 flex items-center justify-between bg-emerald-500/10 border border-emerald-500/30 p-3 rounded-lg backdrop-blur-sm">
                            <div className="flex items-center gap-2">
                                <h3 className="font-bold text-emerald-600 dark:text-emerald-400 text-sm uppercase tracking-wider">High Probability (CIS)</h3>
                            </div>
                            <span className="text-xs font-mono text-emerald-600/70">Low Friction Entry</span>
                        </div>
              {companies.filter(c => c.segment === 'Global CIS').map(company => (
                            <CompanyCard 
                                key={company.name} 
                                company={company} 
                                variant="cis" 
                                theme={theme} 
                                isSelected={selectedTargets.includes(company.name)}
                                onToggle={toggleTarget}
                            />
              ))}
            </TabsContent>

                    <TabsContent value="list-b" className="pr-2 space-y-4 data-[state=inactive]:hidden">
                        <div className="mb-4 flex items-center justify-between bg-purple-500/10 border border-purple-500/30 p-3 rounded-lg backdrop-blur-sm">
                            <div className="flex items-center gap-2">
                                <h3 className="font-bold text-purple-600 dark:text-purple-400 text-sm uppercase tracking-wider">High Salary (Enterprise)</h3>
                            </div>
                            <span className="text-xs font-mono text-purple-600/70">Max Comp Potential</span>
                        </div>
              {companies.filter(c => c.segment !== 'Global CIS').map(company => (
                            <CompanyCard 
                                key={company.name} 
                                company={company} 
                                variant="enterprise" 
                                theme={theme} 
                                isSelected={selectedTargets.includes(company.name)}
                                onToggle={toggleTarget}
                            />
                        ))}
                    </TabsContent>
                </Tabs>
            </TabsContent>

            <TabsContent value="market-pulse" className="pr-2 space-y-6 data-[state=inactive]:hidden animate-in fade-in slide-in-from-bottom-4">
                <MarketAnalysisTab theme={theme} />
            </TabsContent>

            <TabsContent value="company-manager" className="pr-2 space-y-6 data-[state=inactive]:hidden animate-in fade-in slide-in-from-bottom-4">
                <CompanyManagerTab theme={theme} />
            </TabsContent>

            <TabsContent value="persona" className="pr-2 space-y-6 data-[state=inactive]:hidden animate-in fade-in slide-in-from-bottom-4">
                <PersonaOptimizerTab idealPersona={idealPersona} userBackground={userBackground} theme={theme} />
                <LinkedInPostGenerator theme={theme} />
            </TabsContent>

            <TabsContent value="resume-ops" className="pr-2 space-y-6 data-[state=inactive]:hidden animate-in fade-in slide-in-from-bottom-4">
                <ResumeOpsTab theme={theme} />
            </TabsContent>

            <TabsContent value="voice-sim" className="pr-2 space-y-6 data-[state=inactive]:hidden animate-in fade-in slide-in-from-bottom-4">
                <VoiceSimTab theme={theme} />
            </TabsContent>
          </Tabs>
        </div>

        {/* Right Column: Generator & Stats */}
        <div className="space-y-6 md:space-y-8 pl-2">
          <B1GeneratorCard onOpen={() => setIsGeneratorOpen(true)} theme={theme} />
          <TrojanHorseCard onOpen={() => setIsTrojanOpen(true)} theme={theme} />

          <Card className="border-border bg-card/50 backdrop-blur-md">
            <CardHeader>
              <CardTitle className="font-mono uppercase text-sm tracking-widest text-muted-foreground">Mission_Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 font-mono">
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded border border-border">
                  <span className="text-xs md:text-sm text-muted-foreground">TARGETS_LOCKED</span>
                  <span className="text-lg md:text-xl font-bold text-primary">{companies.length}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded border border-border">
                  <span className="text-xs md:text-sm text-muted-foreground">PRIORITY_TARGETS</span>
                  <span className="text-lg md:text-xl font-bold text-yellow-500">{selectedTargets.length}</span>
                </div>
                
                {/* Pipeline Status */}
                <div className="p-3 bg-muted/50 rounded border border-border space-y-2">
                    <div className="flex justify-between items-center">
                        <span className="text-xs md:text-sm text-muted-foreground">DATA_FRESHNESS</span>
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isSyncing ? 'bg-yellow-500 animate-ping' : 'bg-green-500 animate-pulse'}`}></div>
                            <span className={`text-[10px] font-bold ${isSyncing ? 'text-yellow-500' : 'text-green-500'}`}>
                                {isSyncing ? 'SYNCING...' : 'LIVE'}
                            </span>
                        </div>
                    </div>
                    <div className="text-[10px] text-muted-foreground flex justify-between items-center">
                        <span>Last Sync:</span>
                        <span className="font-mono text-foreground">
                            {companies[0]?.last_updated 
                                ? new Date(companies[0].last_updated).toLocaleTimeString() 
                                : "Manual Sync Req."}
                        </span>
                    </div>
                    <Button 
                        variant="outline" 
                        size="sm" 
                        className={`w-full h-6 text-[10px] uppercase tracking-widest border-dashed ${isSyncing ? 'bg-muted' : 'hover:bg-primary/10 hover:border-primary'}`}
                        onClick={async () => {
                            setIsSyncing(true);
                            setSyncResult(null);
                            try {
                                const res = await fetch(`${API_URL}/api/trigger-pipeline`, { method: 'POST' });
                                const data = await res.json();
                                if (data.success) {
                                    console.log(data.logs);
                                    // Parse logs to extract count
                                    const match = data.logs.match(/Database updated with (\d+) new validated roles/);
                                    const count = match ? parseInt(match[1]) : 0;
                                    setSyncResult({ 
                                        count, 
                                        message: count > 0 ? `${count} new vacancies added!` : 'No new vacancies found.' 
                                    });
                                    setTimeout(() => window.location.reload(), 2000);
                                } else {
                                    alert("Sync Failed: " + data.error);
                                }
                            } catch (e) {
                                alert("Backend Offline. Please run 'node server.js' in the terminal.");
                            } finally {
                                setIsSyncing(false);
                            }
                        }}
                        disabled={isSyncing}
                    >
                        <RefreshCw className={`w-3 h-3 mr-2 ${isSyncing ? 'animate-spin' : ''}`} /> 
                        {isSyncing ? "Enriching_Data..." : "Sync_Intel_Network"}
                    </Button>
                    {syncResult && (
                        <div className={`mt-2 p-2 rounded text-[10px] font-mono ${
                            syncResult.count > 0 
                                ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                                : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                        }`}>
                            {syncResult.message}
                        </div>
                    )}
                </div>

                <div className="border-t border-border pt-4 md:pt-6">
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-bold mb-3">
                        SKILL_MATRIX_MATCH
                        <span className="ml-2 text-cyan-500">(Top Skills)</span>
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {userProfile && userProfile.core_ai_skills ? (
                            <>
                                {userProfile.core_ai_skills.slice(0, 2).map((skill: string, idx: number) => (
                                    <Badge key={`core-${idx}`} variant="outline" className="border-cyan-500/50 text-cyan-500 bg-cyan-500/10 font-mono text-[10px]">
                                        {skill}
                                    </Badge>
                                ))}
                                {userProfile.technical_skills?.slice(0, 1).map((skill: string, idx: number) => (
                                    <Badge key={`tech-${idx}`} variant="outline" className="border-purple-500/50 text-purple-500 bg-purple-500/10 font-mono text-[10px]">
                                        {skill}
                                    </Badge>
                                ))}
                                {userProfile.business_skills?.slice(0, 1).map((skill: string, idx: number) => (
                                    <Badge key={`biz-${idx}`} variant="outline" className="border-orange-500/50 text-orange-500 bg-orange-500/10 font-mono text-[10px]">
                                        {skill}
                                    </Badge>
                                ))}
                            </>
                        ) : (
                            <span className="text-xs text-muted-foreground italic">
                                Loading skills... 
                                <span className="ml-2 text-[10px]">(Go to Dashboard to edit profile)</span>
                            </span>
                        )}
                    </div>
                    <p className="text-[9px] text-muted-foreground mt-2 font-mono">
                        💡 Edit full profile above to update AI matching
                    </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
      
      <GeneratorDialog open={isGeneratorOpen} onOpenChange={setIsGeneratorOpen} theme={theme} />
      <TrojanHorseDialog open={isTrojanOpen} onOpenChange={setIsTrojanOpen} theme={theme} />
    </div>
  );
}

function PersonaOptimizerTab({ idealPersona, userBackground, theme }: { idealPersona: any, userBackground: any, theme: string }) {
    const [introScript, setIntroScript] = useState("");
    const [phrases, setPhrases] = useState("");
    const [personaAnalysis, setPersonaAnalysis] = useState("");
    const [selectedCompany, setSelectedCompany] = useState(companies[0]);
    const [loadingIntro, setLoadingIntro] = useState(false);
    const [loadingPhrases, setLoadingPhrases] = useState(false);
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    const [isPolyglotOpen, setIsPolyglotOpen] = useState(false);
    const { sentencesCompleted, mistakeCount, isPassed } = useLessonStore();

    const handleGenerateIntro = async () => {
        setLoadingIntro(true);
        const text = await generatePersonaIntro(idealPersona, userBackground);
        setIntroScript(text);
        setLoadingIntro(false);
    };

    const handleGeneratePhrases = async (category: string) => {
        setLoadingPhrases(true);
        const text = await generateInterviewPhrases(category);
        setPhrases(text);
        setLoadingPhrases(false);
    };

    const handleAnalyzePersona = async () => {
        setLoadingAnalysis(true);
        const role = selectedCompany.open_roles[0]?.title || "Product Manager";
        const text = await generatePersonaAnalysis(selectedCompany.name, role);
        setPersonaAnalysis(text);
        setLoadingAnalysis(false);
    };

    return (
        <div className="space-y-6">
            {/* Polyglot English Trainer Card */}
            <Card className="border-l-4 border-l-cyan-500 bg-gradient-to-r from-cyan-500/5 to-transparent shadow-lg relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
                <CardHeader>
                    <CardTitle className="text-cyan-500 flex items-center gap-2 uppercase tracking-wider text-sm">
                        <BookOpen className="w-4 h-4" /> Polyglot_English_Trainer
                    </CardTitle>
                    <CardDescription className="font-mono text-xs">
                    Interactive simulator of English grammar according to Petrov's method
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-muted/30 rounded border border-border text-center">
                            <div className="text-xl font-black text-cyan-500">{sentencesCompleted}/100</div>
                            <div className="text-[8px] uppercase tracking-widest text-muted-foreground">прогресс</div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded border border-border text-center">
                            <div className="text-xl font-black text-cyan-500">3</div>
                            <div className="text-[8px] uppercase tracking-widest text-muted-foreground">времени</div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded border border-border text-center">
                            <div className={`text-xl font-black ${isPassed() ? 'text-emerald-500' : 'text-cyan-500'}`}>
                                {isPassed() ? 'PASS' : 'B1'}
                            </div>
                            <div className="text-[8px] uppercase tracking-widest text-muted-foreground">уровень</div>
                        </div>
                    </div>
                    <p className="text-xs font-mono text-muted-foreground leading-relaxed">
                        &gt; TRAINING_MODE: Sentence_Builder<br />
                        &gt; GRAMMAR_FOCUS: Simple_Tenses<br />
                        &gt; STATUS: READY_TO_LAUNCH
                    </p>
                    <Button 
                        onClick={() => setIsPolyglotOpen(true)} 
                        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-bold font-mono uppercase tracking-widest text-xs h-12 shadow-[0_0_20px_rgba(8,145,178,0.3)] hover:shadow-[0_0_30px_rgba(8,145,178,0.5)] transition-all duration-300"
                    >
                        <Zap className="mr-2 h-4 w-4" /> Launch_English_Trainer
                    </Button>
                </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="border-border bg-card/50">
            <CardHeader>
                        <CardTitle className="text-primary flex items-center gap-2 uppercase tracking-wider text-sm">
                            <UserCheck className="w-4 h-4" /> Market Persona
              </CardTitle>
            </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="bg-muted/30 p-3 rounded border border-border">
                            <h4 className="text-[10px] uppercase font-bold text-muted-foreground mb-2">Market Demands</h4>
                            <div className="flex flex-wrap gap-2">
                                {idealPersona.top_skills.map((s: string) => <Badge key={s} variant="secondary" className="text-[10px]">{s}</Badge>)}
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-border bg-card/50 flex flex-col">
                    <CardHeader>
                        <CardTitle className="text-foreground flex items-center gap-2 uppercase tracking-wider text-sm">
                            <BookOpen className="w-4 h-4" /> Company Archetype
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col gap-4">
                        <div className="flex gap-2">
                            <select 
                                className="flex-1 bg-background border border-input rounded text-xs p-2"
                                value={selectedCompany.name}
                                onChange={(e) => setSelectedCompany(companies.find(c => c.name === e.target.value) || companies[0])}
                            >
                                {companies.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                            </select>
                            <Button size="sm" onClick={handleAnalyzePersona} disabled={loadingAnalysis} className="relative overflow-hidden">
                                {loadingAnalysis ? (
                                    <>
                                        <div className="absolute inset-0 bg-gradient-to-r from-primary via-purple-500 to-primary animate-pulse" />
                                        <span className="relative z-10">
                                            <Loader2 className="w-3 h-3 animate-spin" />
                                        </span>
                                    </>
                                ) : (
                                    "Analyze"
                                )}
                            </Button>
                        </div>
                        {loadingAnalysis && (
                            <div className="flex items-center justify-center gap-1 py-2">
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                            </div>
                        )}
                        {personaAnalysis && (
                            <AIContentViewer 
                                content={personaAnalysis} 
                                title={`Persona Analysis - ${selectedCompany.name}`}
                            />
                        )}
                    </CardContent>
                </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Intro Generator */}
                <Card className="border-border bg-card/50 flex flex-col">
                    <CardHeader>
                        <CardTitle className="text-foreground flex items-center gap-2 uppercase tracking-wider text-sm">
                            <Mic className="w-4 h-4" /> "Tell Me About Yourself"
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col gap-4">
                        <p className="text-xs text-muted-foreground">Generate a B1-friendly intro script.</p>
                        <Button onClick={handleGenerateIntro} disabled={loadingIntro} className="w-full bg-primary hover:bg-primary/90 text-primary-foreground relative overflow-hidden">
                            {loadingIntro ? (
                                <>
                                    <div className="absolute inset-0 bg-gradient-to-r from-primary via-blue-500 to-primary animate-pulse" />
                                    <span className="relative z-10 flex items-center justify-center">
                                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                                        Generating...
                                    </span>
                                </>
                            ) : (
                                "Generate Script"
                            )}
                        </Button>
                        {loadingIntro && (
                            <div className="flex items-center justify-center gap-1">
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                            </div>
                        )}
                        {introScript && (
                            <AIContentViewer 
                                content={introScript} 
                                title="Interview Introduction Script"
                            />
                        )}
                    </CardContent>
                </Card>

                {/* Phrase Practice */}
                <Card className="border-border bg-card/50 flex flex-col">
                    <CardHeader>
                        <CardTitle className="text-foreground flex items-center gap-2 uppercase tracking-wider text-sm">
                            <BookOpen className="w-4 h-4" /> Phrasebook
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col gap-4">
                        <p className="text-xs text-muted-foreground">Practice key phrases.</p>
                        <div className="grid grid-cols-2 gap-2">
                            <Button variant="outline" size="sm" onClick={() => handleGeneratePhrases("Technical Architecture")} disabled={loadingPhrases} className="text-[10px]">Tech Arch</Button>
                            <Button variant="outline" size="sm" onClick={() => handleGeneratePhrases("Conflict Resolution")} disabled={loadingPhrases} className="text-[10px]">Conflict</Button>
                            <Button variant="outline" size="sm" onClick={() => handleGeneratePhrases("Prioritization")} disabled={loadingPhrases} className="text-[10px]">Prioritization</Button>
                            <Button variant="outline" size="sm" onClick={() => handleGeneratePhrases("Stakeholder Mgmt")} disabled={loadingPhrases} className="text-[10px]">Stakeholders</Button>
                        </div>
                        {loadingPhrases && (
                            <div className="flex items-center justify-center gap-1 py-2">
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                <div className="h-1 w-1 bg-primary rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                            </div>
                        )}
                        {phrases && (
                            <AIContentViewer 
                                content={phrases} 
                                title="Interview Phrases Practice"
                            />
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Polyglot Modal */}
            <PolyglotModal 
                open={isPolyglotOpen} 
                onOpenChange={setIsPolyglotOpen} 
                theme={theme} 
            />
        </div>
    )
}

function VoiceSimTab({ theme }: { theme: string }) {
    const [isListening, setIsListening] = useState(false);
    const [chatHistory, setChatHistory] = useState<{ role: "user" | "model", content: string }[]>([]);
    const [feedback, setFeedback] = useState("");
    const [processing, setProcessing] = useState(false);

    // Web Speech API Setup
    // @ts-ignore
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
    }

    const startListening = () => {
        if (!recognition) {
            alert("Web Speech API not supported in this browser.");
            return;
        }
        setIsListening(true);
        recognition.start();

        recognition.onresult = async (event: any) => {
            const transcript = event.results[0][0].transcript;
            console.log("User said:", transcript);
            setIsListening(false);
            
            // Add user input to history
            const newHistory = [...chatHistory, { role: "user" as const, content: transcript }];
            setChatHistory(newHistory);
            setProcessing(true);

            // Generate AI response
            const aiResponse = await import('./lib/gemini').then(mod => mod.generateVoiceSimResponse(newHistory, transcript));
            
            // Extract potential feedback from response if structured, else just show raw
            setChatHistory([...newHistory, { role: "model" as const, content: aiResponse }]);
            setProcessing(false);

            // Simple TTS for AI response
            const utterance = new SpeechSynthesisUtterance(aiResponse);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        };

        recognition.onerror = (event: any) => {
            console.error("Speech recognition error", event.error);
            setIsListening(false);
        };
        
        recognition.onend = () => {
            setIsListening(false);
        };
    };

    const resetSim = () => {
        setChatHistory([]);
        setFeedback("");
        window.speechSynthesis.cancel();
    };

    return (
        <div className="space-y-6">
            <Card className="border-border bg-card/50">
                <CardHeader>
                    <CardTitle className="text-primary flex items-center gap-2 uppercase tracking-wider text-sm">
                        <Mic className="w-4 h-4" /> Voice_Interview_Simulator
                    </CardTitle>
                    <CardDescription className="font-mono text-xs">
                        Practice B1 English with an AI Hiring Manager.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {/* Chat Display */}
                    <div className="h-[300px] overflow-y-auto border border-border rounded bg-background/50 p-4 space-y-4 custom-scrollbar">
                        {chatHistory.length === 0 && (
                            <p className="text-xs text-muted-foreground text-center italic">
                                Press "Start Interview" to begin. The AI will ask you a question.
                            </p>
                        )}
                        {chatHistory.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[80%] p-3 rounded-lg text-xs font-mono ${
                                    msg.role === 'user' 
                                        ? 'bg-primary/20 dark:text-white text-black border border-primary/30' 
                                        : 'bg-muted text-muted-foreground border border-border'
                                }`}>
                                    <span className="block font-bold uppercase text-[8px] mb-1 opacity-70">{msg.role}</span>
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                        {processing && (
                            <div className="flex justify-start">
                                <div className="bg-muted p-3 rounded-lg border border-border flex items-center gap-2">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    <span className="text-xs font-mono">AI_THINKING...</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    <div className="flex justify-center gap-4">
                        {!isListening ? (
                            <Button onClick={startListening} className="bg-red-500 hover:bg-red-600 text-white font-bold font-mono uppercase tracking-widest w-40 h-12 rounded-full shadow-[0_0_15px_rgba(239,68,68,0.5)] animate-pulse">
                                <Mic className="mr-2 h-5 w-5" /> Speak
              </Button>
                        ) : (
                            <Button disabled className="bg-red-500/50 text-white font-bold font-mono uppercase tracking-widest w-40 h-12 rounded-full">
                                <div className="flex gap-1 h-3 items-center">
                                    <span className="w-1 h-full bg-white animate-[bounce_1s_infinite]"></span>
                                    <span className="w-1 h-2/3 bg-white animate-[bounce_1s_infinite_0.2s]"></span>
                                    <span className="w-1 h-full bg-white animate-[bounce_1s_infinite_0.4s]"></span>
                                </div>
                            </Button>
                        )}
                        <Button variant="outline" onClick={resetSim} className="font-mono uppercase text-xs">
                            Reset
                        </Button>
                    </div>
            </CardContent>
          </Card>
        </div>
    )
}

function B1GeneratorCard({ onOpen, theme }: { onOpen: () => void, theme: string }) {
    return (
        <Card className={`border-l-4 shadow-lg ${theme === 'dark' ? 'border-l-yellow-500 bg-yellow-500/5' : 'border-l-yellow-400 bg-yellow-50'}`}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400 uppercase tracking-widest font-bold text-sm">
                        <Sparkles className="h-4 w-4" /> B1_Bridge_Generator
              </CardTitle>
                </div>
                <CardDescription className="font-mono text-xs">
                    Bypass Language Barriers with Vibecoding
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs font-mono text-muted-foreground leading-relaxed">
                    &gt; DETECTED_LANG_BARRIER: B1<br />
                    &gt; INITIATING_PROTOCOL: SIMPLE_AND_PUNCHY<br />
                    &gt; STATUS: READY_TO_GENERATE
                </p>
                <Button onClick={onOpen} className="w-full bg-yellow-500 hover:bg-yellow-400 text-black font-bold font-mono uppercase tracking-widest text-xs">
                    Open_Generator_Module
              </Button>
            </CardContent>
          </Card>
    )
}

function TrojanHorseCard({ onOpen, theme }: { onOpen: () => void, theme: string }) {
    return (
        <Card className={`border-l-4 shadow-lg ${theme === 'dark' ? 'border-l-red-500 bg-red-500/5' : 'border-l-red-600 bg-red-50'}`}>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-red-600 dark:text-red-400 uppercase tracking-widest font-bold text-sm">
                        <FileCode2 className="h-4 w-4" /> Trojan_Horse_Architect
              </CardTitle>
                </div>
                <CardDescription className="font-mono text-xs">
                    Generate High-Level Technical Spec
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs font-mono text-muted-foreground leading-relaxed">
                    &gt; TARGET: TECHNICAL_DECISION_MAKER<br />
                    &gt; PAYLOAD: SYSTEM_ARCHITECTURE_DOC<br />
                    &gt; STATUS: STANDBY
                </p>
                <Button onClick={onOpen} className="w-full bg-red-600 hover:bg-red-500 text-white font-bold font-mono uppercase tracking-widest text-xs">
                    Initialize_Architect
              </Button>
            </CardContent>
          </Card>
    )
}


function GeneratorDialog({ open, onOpenChange, theme }: { open: boolean, onOpenChange: (open: boolean) => void, theme: string }) {
    const [selectedCompany, setSelectedCompany] = useState(companies[0]);
    const [generatedText, setGeneratedText] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);

    const handleGenerate = async () => {
        setIsGenerating(true);
        const painPoints = selectedCompany.strategy_analysis?.pain_points || [];
        const killerFeature = selectedCompany.strategy_analysis?.user_match?.killer_feature || "AI Automation";
        
        const text = await generateCoverLetter(
            selectedCompany.name,
            "Product Manager",
            painPoints,
            ["LangChain", "Voice AI", "Agency Experience", "Vibecoding"],
            killerFeature,
            "professional"
        );
        setGeneratedText(text);
        setIsGenerating(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[600px] font-mono">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 uppercase tracking-widest text-yellow-500">
                        <Sparkles className="h-5 w-5" /> B1_Bridge_Generator
                    </DialogTitle>
                    <DialogDescription>
                        Select a target and generate a simple, punchy cover letter.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <label htmlFor="company" className="text-right text-xs font-bold uppercase text-muted-foreground">
                            Target
                        </label>
                        <div className="col-span-3">
                        <select 
                            id="company" 
                                className="w-full p-2 border border-input bg-background rounded text-sm"
                            value={selectedCompany.name}
                            onChange={(e) => setSelectedCompany(companies.find(c => c.name === e.target.value) || companies[0])}
                        >
                                {companies.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                        </select>
                    </div>
                    </div>
                    <div className="col-span-4">
                         <Button onClick={handleGenerate} disabled={isGenerating} className="w-full bg-yellow-500 hover:bg-yellow-400 text-black font-bold uppercase tracking-widest relative overflow-hidden">
                            {isGenerating ? (
                                <>
                                    <div className="absolute inset-0 bg-gradient-to-r from-yellow-500 via-orange-400 to-yellow-500 animate-pulse" />
                                    <span className="relative z-10 flex items-center justify-center">
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Compiling_Payload...
                                    </span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Generate_Transmission
                                </>
                            )}
                        </Button>
                            </div>
                    {isGenerating && (
                        <div className="col-span-4 mt-3 text-center">
                            <div className="flex items-center justify-center gap-1 mb-2">
                                <div className="h-1.5 w-1.5 bg-yellow-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                <div className="h-1.5 w-1.5 bg-yellow-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                <div className="h-1.5 w-1.5 bg-yellow-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                        </div>
                            <p className="text-[10px] text-yellow-600 font-mono animate-pulse">
                                ✨ Crafting personalized cover letter...
                            </p>
                </div>
                    )}
                    {generatedText && (
                        <div className="col-span-4 mt-4">
                             <label className="text-xs font-bold uppercase text-muted-foreground block mb-2">Generated Output</label>
                             <AIContentViewer 
                                content={generatedText} 
                                title={`Cover Letter - ${selectedCompany.name}`}
                            />
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    )
}

function TrojanHorseDialog({ open, onOpenChange, theme }: { open: boolean, onOpenChange: (open: boolean) => void, theme: string }) {
    const [selectedCompany, setSelectedCompany] = useState(companies[0]);
    const [generatedStrategy, setGeneratedStrategy] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);

    const handleGenerate = async () => {
        setIsGenerating(true);
        const painPoints = selectedCompany.strategy_analysis?.pain_points || [];
        const killerFeature = selectedCompany.strategy_analysis?.user_match?.killer_feature || "AI Automation System";
        
        const text = await generateTrojanHorseStrategy(
            selectedCompany.name,
            killerFeature,
            painPoints
        );
        setGeneratedStrategy(text);
        setIsGenerating(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[700px] font-mono h-[80vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 uppercase tracking-widest text-red-500">
                        <FileCode2 className="h-5 w-5" /> Trojan_Horse_Architect
                    </DialogTitle>
                    <DialogDescription>
                        Generate a technical specification and architecture document to impress the hiring manager.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4 flex-1 overflow-hidden flex flex-col">
                    <div className="grid grid-cols-4 items-center gap-4 shrink-0">
                        <label htmlFor="company-trojan" className="text-right text-xs font-bold uppercase text-muted-foreground">
                            Target
                        </label>
                        <div className="col-span-3">
                        <select 
                            id="company-trojan" 
                                className="w-full p-2 border border-input bg-background rounded text-sm"
                            value={selectedCompany.name}
                            onChange={(e) => setSelectedCompany(companies.find(c => c.name === e.target.value) || companies[0])}
                        >
                                {companies.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                        </select>
                    </div>
                    </div>
                    <div className="col-span-4 shrink-0">
                         <Button onClick={handleGenerate} disabled={isGenerating} className="w-full bg-red-600 hover:bg-red-500 text-white font-bold uppercase tracking-widest relative overflow-hidden">
                            {isGenerating ? (
                                <>
                                    <div className="absolute inset-0 bg-gradient-to-r from-red-600 via-orange-500 to-red-600 animate-pulse" />
                                    <span className="relative z-10 flex items-center justify-center">
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Architecting_Solution...
                                    </span>
                                </>
                            ) : (
                                <>
                                    <FileCode2 className="mr-2 h-4 w-4" />
                                    Generate_Technical_Spec
                                </>
                            )}
                    </Button>
                        {isGenerating && (
                            <div className="mt-3 text-center">
                                <div className="flex items-center justify-center gap-1 mb-2">
                                    <div className="h-1.5 w-1.5 bg-red-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                    <div className="h-1.5 w-1.5 bg-red-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                    <div className="h-1.5 w-1.5 bg-red-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                    </div>
                                <p className="text-[10px] text-red-500 font-mono animate-pulse">
                                    🛠️ Designing technical architecture...
                                </p>
                    </div>
                        )}
                            </div>
                    {generatedStrategy && (
                        <div className="col-span-4 mt-4 flex-1 overflow-hidden flex flex-col">
                             <label className="text-xs font-bold uppercase text-muted-foreground block mb-2 shrink-0">Generated Specification</label>
                             <AIContentViewer 
                                content={generatedStrategy} 
                                title={`Technical Specification - ${selectedCompany.name}`}
                                className="flex-1"
                            />
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    )
}

function CompanyCard({ company, variant, theme, isSelected = false, onToggle }: { company: any, variant: 'cis' | 'enterprise', theme: string, isSelected?: boolean, onToggle?: (name: string) => void }) {
  const [isOpen, setIsOpen] = useState(false);
  
  // Dynamic styling based on theme and variant
  const isDark = theme === 'dark';
  
  const textPrimary = variant === 'cis' 
    ? (isDark ? 'text-green-400' : 'text-green-600') 
    : (isDark ? 'text-purple-400' : 'text-purple-600');

  const borderHover = variant === 'cis'
    ? (isDark ? 'hover:border-green-500' : 'hover:border-green-600')
    : (isDark ? 'hover:border-purple-500' : 'hover:border-purple-600');

  const badgeStrategy = variant === 'cis'
    ? (isDark ? 'border-green-900 bg-green-950 text-green-400' : 'border-green-200 bg-green-100 text-green-800')
    : (isDark ? 'border-purple-900 bg-purple-950 text-purple-400' : 'border-purple-200 bg-purple-100 text-purple-800');

  const deepDiveBg = variant === 'cis' 
    ? (isDark ? 'bg-green-950/10 border-green-900/50' : 'bg-green-50 border-green-200')
    : (isDark ? 'bg-purple-950/10 border-purple-900/50' : 'bg-purple-50 border-purple-200');

  const retroIcon = getRandomRetroIcon(company.name);
  const hasNewRoles = company.open_roles?.some((r: any) => r.is_new);

  return (
    <Card className={`transition-all duration-300 hover:shadow-lg group ${borderHover} ${isSelected ? 'border-yellow-500/50 shadow-yellow-500/10' : ''} ${hasNewRoles ? 'shadow-[0_0_20px_rgba(34,197,94,0.2)] border-green-500/50' : ''}`}>
      <CardContent className="p-0">
        {/* Habr-style Retro Header */}
        <div className="h-24 bg-muted/80 border-b border-border flex items-center justify-between px-6 relative overflow-hidden">
            <div className="absolute inset-0 opacity-10 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiMwMDAiLz48L3N2Zz4=')] dark:bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiNmZmYiLz48L3N2Zz4=')]"></div>
            <div className="flex items-center gap-4 relative z-10">
                <div className="w-16 h-16 bg-zinc-200 dark:bg-zinc-800 border-2 border-zinc-300 dark:border-zinc-700 flex items-center justify-center text-4xl rounded-lg shadow-inner relative">
                    {retroIcon}
                    {hasNewRoles && <span className="absolute -top-2 -right-2 flex h-4 w-4"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span><span className="relative inline-flex rounded-full h-4 w-4 bg-red-500"></span></span>}
                </div>
                <div className="flex flex-col justify-center">
                    <h3 className="text-xl md:text-2xl font-black tracking-tighter flex items-center gap-2">
                        {company.name}
                        {isSelected && <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />}
                        {hasNewRoles && <span className="animate-pulse text-[10px] font-bold px-2 py-0.5 rounded bg-red-500 text-white shadow-[0_0_10px_rgba(239,68,68,0.5)] border border-red-400">NEW_INTEL</span>}
            </h3>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono uppercase tracking-wider">
                        <span>{company.size}</span>
                        <span className="text-border">|</span>
                        <span>{company.funding}</span>
            </div>
          </div>
            </div>
            {onToggle && (
                <Button 
                    variant="ghost" 
                    size="icon" 
                    className={`relative z-10 h-10 w-10 rounded-full bg-background/50 border border-border hover:bg-background ${isSelected ? 'text-yellow-500 hover:text-yellow-600' : 'text-muted-foreground hover:text-foreground'}`}
                    onClick={(e) => { e.stopPropagation(); onToggle(company.name); }}
                >
                    <Star className={`w-5 h-5 ${isSelected ? 'fill-yellow-500' : ''}`} />
                </Button>
            )}
        </div>

        <div className="p-4 md:p-6 pt-4">
            <div className="flex justify-between items-start mb-4">
                <div className="space-y-2">
                    {company.strategy_analysis && <span className={`text-[10px] font-mono px-2 py-0.5 rounded border w-fit ${badgeStrategy}`}>STRATEGY_READY</span>}
                </div>
          <div className="text-right shrink-0 ml-4">
            <div className={`text-3xl md:text-4xl font-black tracking-tighter ${textPrimary}`}>
                {company.metrics.hiring_probability}%
            </div>
            <div className="text-[8px] md:text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Match_Prob</div>
          </div>
        </div>
        
        <div className="space-y-4">
          {/* Primary Action Item - Always Visible */}
          <div className={`p-4 rounded border border-dashed ${deepDiveBg} flex gap-3 items-start`}>
            <Zap className={`h-5 w-5 ${textPrimary} shrink-0 mt-0.5`} />
            <div>
                <h4 className={`font-bold ${textPrimary} text-xs uppercase tracking-wider mb-1`}>Secret_Weapon_Strategy</h4>
                <p className="text-xs md:text-sm text-foreground font-medium leading-relaxed">
                    {company.strategy_analysis?.user_match?.killer_feature || company.action_item}
                </p>
            </div>
          </div>

          {/* Expandable Deep Dive */}
          {isOpen && company.strategy_analysis && (
            <div className="mt-6 space-y-6 border-t border-border pt-6 animate-in fade-in slide-in-from-top-2 duration-300">
                
                {/* 1. Pain Points Analysis */}
                <div>
                    <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                        <AlertTriangle className="h-3 w-3" /> System_Vulnerabilities (Pain Points)
                    </h4>
                    <ul className="grid gap-2">
                        {company.strategy_analysis.pain_points.map((point: string, i: number) => (
                            <li key={i} className="text-sm bg-destructive/10 text-destructive-foreground px-3 py-2 rounded border border-destructive/20 flex gap-3 font-mono items-start">
                                <span className="font-bold text-destructive shrink-0">ERROR_{i+1}:</span> <span>{point}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* 2. Match Analysis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Your_Modules (Strengths)</h4>
                        <div className="flex flex-wrap gap-2">
                            {company.strategy_analysis.user_match.strengths.map((s: string) => (
                                <Badge key={s} variant="outline" className="bg-primary/10 text-primary border-primary/30 font-mono text-[10px]">{s}</Badge>
                            ))}
                        </div>
                    </div>
                    <div>
                        <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Missing_Dependencies (Gaps)</h4>
                        <div className="flex flex-wrap gap-2">
                            {company.strategy_analysis.user_match.gaps.map((g: string) => (
                                <Badge key={g} variant="outline" className="bg-muted text-muted-foreground border-border border-dashed font-mono text-[10px]">{g}</Badge>
                            ))}
                        </div>
                    </div>
                </div>

                {/* 3. Ice Breaker */}
                <div className="bg-blue-500/10 p-4 rounded border border-blue-500/20 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                    <h4 className="text-xs font-bold text-blue-500 mb-2 flex items-center gap-2 uppercase tracking-widest">
                        <MessageSquare className="h-3 w-3" /> Outreach_Payload
                    </h4>
                    <p className="text-sm text-foreground font-mono bg-background/50 p-3 rounded border border-border/50">
                        "{company.strategy_analysis.ice_breaker}"
                    </p>
                    <div className="mt-3 flex gap-2">
                        <Button size="sm" variant="default" className="bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase tracking-widest">Copy_Payload</Button>
                    </div>
                </div>

                {/* 4. Roles */}
                <div>
                        <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">
                            Open_Ports (Roles) 
                            <Badge variant="outline" className="ml-2 text-[8px]">{company.open_roles.length}</Badge>
                        </h4>
                        <div className="space-y-2 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
                            {company.open_roles.map((role: any, idx: number) => (
                                <div key={`${role.url || role.title}-${idx}`} className={`flex flex-col text-sm p-3 border transition-colors rounded font-mono text-foreground ${role.is_new ? 'bg-red-500/10 border-red-500/50 hover:border-red-500' : 'bg-muted/30 border-border hover:border-primary/50'}`}>
                                    <div className="flex items-start justify-between gap-2 mb-2">
                                        <span className="font-medium flex items-center gap-2 flex-1">
                                            {role.is_new && <Badge variant="destructive" className="text-[8px] h-4 px-1">NEW</Badge>}
                                            {role.title}
                                        </span>
                                        {role.ai_analysis?.score && (
                                            <Badge variant="outline" className="text-[8px] bg-green-500/20 border-green-500/50 text-green-400">
                                                AI: {role.ai_analysis.score}%
                                            </Badge>
                                        )}
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground">
                                        <span>📍 {role.location}</span>
                                        {role.applicants && <span>• 👥 {role.applicants}</span>}
                                        {role.url && (
                                            <a 
                                                href={role.url} 
                                                target="_blank" 
                                                rel="noopener noreferrer"
                                                className="text-primary hover:underline"
                                            >
                                                🔗 Link
                                            </a>
                                        )}
                                    </div>
                                    {role.ai_analysis?.reason && (
                                        <p className="text-[10px] text-muted-foreground mt-2 italic border-l-2 border-primary/30 pl-2">
                                            💡 {role.ai_analysis.reason}
                                        </p>
                                    )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
          )}

          <div 
            className={`flex items-center justify-center pt-4 text-[10px] uppercase tracking-widest cursor-pointer transition-colors ${textPrimary} hover:opacity-80`}
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? (
                <span className="flex items-center gap-2">Collapse_Module <ChevronUp className="h-3 w-3" /></span>
            ) : (
                <span className="flex items-center gap-2">Access_Deep_Dive <ChevronDown className="h-3 w-3" /></span>
            )}
          </div>
            </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ResumeOpsTab({ theme }: { theme: string }) {
    const [baseResume, setBaseResume] = useState(`User Background:
- **Role:** AI Product Manager / Advisor / Project Manager.
- **Core AI Skills:** Building AI pipelines (LangChain), LLM, TTS (Text-to-Speech), STT (Speech-to-Text), Realtime Voice Assistants.
- **Technical Skills (Vibecoding):** Python, React, React Native, Electron, Tailwind CSS. No formal CS degree but confident practical usage.
- **Business & Management:**
  - 15 years running an Advertising Agency.
  - SEO, SERM, PR, Marketing.
  - Programmatic SEO, Parsing.
  - Web Development management, Corporate CRM-like systems.
  - Startup creation experience.
- **Target Role:** AI Product Manager in a startup or company implementing AI pipelines.
- **Unique Value Proposition:** Bridge between technical AI implementation (LangChain/Voice) and business growth (SEO/Marketing/Agency background). Understands "vibecoding" and can prototype.`);
    const [referenceResume, setReferenceResume] = useState("");
    const [selectedCompany, setSelectedCompany] = useState(companies[0]);
    const [generatedResume, setGeneratedResume] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [savedResumes, setSavedResumes] = useState<{name: string, content: string}[]>(() => {
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('savedResumes');
            return saved ? JSON.parse(saved) : [];
        }
        return [];
    });

    useEffect(() => {
        localStorage.setItem('savedResumes', JSON.stringify(savedResumes));
    }, [savedResumes]);

    const handleGenerate = async () => {
        setIsGenerating(true);
        const result = await generateTailoredResume(baseResume, selectedCompany.name, selectedCompany, referenceResume || null);
        setGeneratedResume(result);
        setIsGenerating(false);
    };

    const handleSave = () => {
        const name = `${selectedCompany.name}_${new Date().toLocaleDateString()}_${Math.floor(Math.random() * 100)}`;
        setSavedResumes([...savedResumes, { name, content: generatedResume }]);
    };

    return (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
            <div className="space-y-6 flex flex-col h-full">
                <Card className="border-border bg-card/50 flex-1 flex flex-col">
                    <CardHeader>
                         <CardTitle className="text-primary flex items-center gap-2 uppercase tracking-wider text-sm">
                            <FileText className="w-4 h-4" /> Source_Material
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col gap-4 overflow-y-auto custom-scrollbar">
                        <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase text-muted-foreground">Base Resume (Markdown)</label>
                            <textarea 
                                className="w-full h-48 p-3 bg-muted/50 border border-input rounded text-xs font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary"
                                value={baseResume}
                                onChange={(e) => setBaseResume(e.target.value)}
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase text-muted-foreground">Reference Style (Optional)</label>
                            <textarea 
                                className="w-full h-32 p-3 bg-muted/50 border border-input rounded text-xs font-mono resize-none focus:outline-none focus:ring-1 focus:ring-primary"
                                placeholder="Paste a resume here to copy its style/formatting..."
                                value={referenceResume}
                                onChange={(e) => setReferenceResume(e.target.value)}
                            />
                        </div>
                         <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase text-muted-foreground">Target Vector</label>
                            <select 
                                className="w-full p-2 border border-input bg-background rounded text-xs font-mono"
                                value={selectedCompany.name}
                                onChange={(e) => setSelectedCompany(companies.find(c => c.name === e.target.value) || companies[0])}
                            >
                                {companies.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                            </select>
                        </div>
                        <Button onClick={handleGenerate} disabled={isGenerating} className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-bold uppercase tracking-widest text-xs relative overflow-hidden">
                            {isGenerating ? (
                                <>
                                    <div className="absolute inset-0 bg-gradient-to-r from-primary via-cyan-500 to-primary animate-pulse" />
                                    <span className="relative z-10 flex items-center justify-center">
                                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                                        AI Tailoring Resume...
                                    </span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-4 h-4 mr-2" />
                                    Generate_Tailored_Resume
                                </>
                            )}
                        </Button>
                        
                        {isGenerating && (
                            <div className="mt-3 text-center">
                                <div className="flex items-center justify-center gap-1 mb-2">
                                    <div className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                                    <div className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                                    <div className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
                                </div>
                                <p className="text-[10px] text-primary font-mono animate-pulse">
                                    ✨ Analyzing job requirements & matching experience...
                                </p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>

            <div className="space-y-6 flex flex-col h-full">
                 <Card className="border-border bg-card/50 flex-1 flex flex-col h-full">
                    <CardHeader className="flex flex-row items-center justify-between">
                         <CardTitle className="text-primary flex items-center gap-2 uppercase tracking-wider text-sm">
                            <FileCode2 className="w-4 h-4" /> Compile_Output
                        </CardTitle>
                        <div className="flex gap-2">
                            {generatedResume && (
                                <>
                                    <Button size="sm" variant="outline" onClick={handleSave}><Save className="w-3 h-3" /></Button>
                                    <Button size="sm" variant="outline" onClick={() => navigator.clipboard.writeText(generatedResume)}><Copy className="w-3 h-3" /></Button>
                                </>
                            )}
                        </div>
                    </CardHeader>
                    <CardContent className="flex-1 flex flex-col overflow-hidden">
                        {generatedResume ? (
                            <AIContentViewer 
                                content={generatedResume} 
                                title={`Tailored Resume - ${selectedCompany.name}`}
                                className="flex-1"
                            />
                        ) : (
                            <div className="flex-1 flex items-center justify-center text-muted-foreground text-xs font-mono border border-dashed border-border rounded bg-muted/20">
                                WAITING_FOR_COMPILATION...
                            </div>
                        )}
                        
                        {savedResumes.length > 0 && (
                            <div className="mt-4 pt-4 border-t border-border">
                                <label className="text-[10px] font-bold uppercase text-muted-foreground block mb-2">Version_Control</label>
                                <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                                    {savedResumes.map((ver, idx) => (
                                        <Badge 
                                            key={idx} 
                                            variant="outline" 
                                            className="cursor-pointer hover:bg-primary/20 whitespace-nowrap"
                                            onClick={() => setGeneratedResume(ver.content)}
                                        >
                                            {ver.name}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
