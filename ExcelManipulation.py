import openpyxl
import xlwings as xw






class PIDSimulator():

    def __init__(self,Simulator="PIDSimulator.xlsx") -> None:
        self.Simulator = "PIDSimulator.xlsx"
        self.Calculation = "Calculation"
        self.LevelTrend = "Level"
        self.Kp = 'J6'
        self.Ti = 'J7'
        self.Td = 'J8'
        self.InitialValveOpening = 'J10'

        self.MaxInflow = 'J17'
        self.MinLevel = 'J18'
        self.MaxLevel = 'J19'
        self.SetPoint = 'J20'
        self.StaticInitialFout = 'J21'
        self.Diameter = 'J22'
        self.Section = 'J23'
        self.Disturbance = 'J24'
        self.DeadTime = 'J25'
        self.workbook = xw.Book('PIDSimulator.xlsx')
        self.TotalError = 'Z2118'
        self.MaxError = 'Z2119'
        self.PVJitter = 'AC2118'
        self.AbsErrorQ2 = 'Z2121'
        self.trailingerror = 'Z2123'
        self.InitialState = []

    def saveInitialState(self):
        self.InitialState = self.getState()


    def SetInitialProcessValues(self,p=10,i=100,d=0,MaxInflow = 50,MinLevel = 0, MaxLevel = 10, SetPoint = 5,
        StaticInitialFout = 20, Diameter = 2.5, Section = 4.9, Disturbance = 15, DeadTime = 1):
        self.workbook.sheets['Calculation'].range(self.MaxInflow).value = MaxInflow
        self.workbook.sheets['Calculation'].range(self.MinLevel).value = MinLevel
        self.workbook.sheets['Calculation'].range(self.MaxLevel).value = MaxLevel
        self.workbook.sheets['Calculation'].range(self.SetPoint).value = SetPoint
        self.workbook.sheets['Calculation'].range(self.StaticInitialFout).value = StaticInitialFout
        self.workbook.sheets['Calculation'].range(self.Diameter).value = Diameter
        self.workbook.sheets['Calculation'].range(self.Section).value = Section
        self.workbook.sheets['Calculation'].range(self.Disturbance).value = Disturbance
        self.workbook.sheets['Calculation'].range(self.DeadTime).value = DeadTime
        self.workbook.sheets['Calculation'].range(self.Kp).value = p
        self.workbook.sheets['Calculation'].range(self.Ti).value = i
        self.workbook.sheets['Calculation'].range(self.Td).value = d

    
    def Simulate(self,PID):

        P,I,D = PID
        # old_state = self.getState()

        self.workbook.sheets['Calculation'].range(self.Kp).value = P if P>1 else 1
        self.workbook.sheets['Calculation'].range(self.Ti).value = I if I>1 else 1
        self.workbook.sheets['Calculation'].range(self.Td).value = D
        

        MaxError = self.workbook.sheets['Calculation'].range(self.MaxError).value
        TotalError = self.workbook.sheets['Calculation'].range(self.TotalError).value
        PVJitter = self.workbook.sheets['Calculation'].range(self.PVJitter).value
        AbsErrorQ2 = self.workbook.sheets['Calculation'].range(self.AbsErrorQ2).value
        AvgTrailError = self.workbook.sheets['Calculation'].range(self.trailingerror).value

        # MaxErrordiv = 1 if MaxError == 0 else MaxError
        # TotalErrordiv = 1 if TotalError == 0 else TotalError
        # PVJitterdiv = 1 if PVJitter == 0 else PVJitter
        # AbsErrorQ2div = 1 if AbsErrorQ2 == 0 else AbsErrorQ2




        # div0 = 1 if self.InitialState[0] == 0 else self.InitialState[0]
        # div1 = 1 if self.InitialState[1] == 0 else self.InitialState[1]
        # div2 = 1 if self.InitialState[2] == 0 else self.InitialState[2]
        # div3 = 1 if self.InitialState[3] == 0 else self.InitialState[3]
        # div4 = 1 if self.InitialState[4] == 0 else self.InitialState[4]

        reward = -TotalError
        



        #reward = 1 - TotalError 
        #reward = (old_state[3] - MaxError)*100/MaxErrordiv + (old_state[4]-TotalError)*100/TotalErrordiv + (old_state[6]-AbsErrorQ2)*100/(AbsErrorQ2div)

        # r1 = 1 if (old_state[0] - MaxError) > 0 else -1
        # r2 = 1 if (old_state[1] - TotalError) > 0 else -1
        # r3 = 1 if (old_state[2] - AbsErrorQ2) > 0 else -1
        # r4 = 1 if (old_state[3] - PVJitter) > 0  else -1

        # r1 = 1 if (self.InitialState[0] - MaxError) > 0 else -1
        # r2 = 1 if (self.InitialState[1] - TotalError) > 0 else -1
        # r3 = 1 if (self.InitialState[2] - AbsErrorQ2) > 0 else -1
        # r4 = 1 if (self.InitialState[3] - PVJitter) > 0  else -1

        # r1 = (self.InitialState[0] - MaxError)/div0
        # r2 = (self.InitialState[1] - TotalError)/div1
        # r3 = (self.InitialState[2] - AbsErrorQ2)/div2
        # r4 = (self.InitialState[3] - PVJitter)/div3
        # r5 = (self.InitialState[4] - AvgTrailError)/div4



        #reward = 0

        self.InitialState[0] = MaxError if MaxError < self.InitialState[0] else self.InitialState[0]
        self.InitialState[1] = TotalError if TotalError < self.InitialState[1] else self.InitialState[1]
        self.InitialState[2] = AbsErrorQ2 if AbsErrorQ2 < self.InitialState[2] else self.InitialState[2]
        self.InitialState[3] = PVJitter if PVJitter < self.InitialState[3] else self.InitialState[3]
        self.InitialState[4] = AvgTrailError if AvgTrailError < self.InitialState[4] else self.InitialState[4]



        

        

        return reward


    def getState(self):

        p = self.workbook.sheets['Calculation'].range(self.Kp).value
        i = self.workbook.sheets['Calculation'].range(self.Ti).value
        #d = self.workbook.sheets['Calculation'].range(self.Td).value
        
        MaxError = self.workbook.sheets['Calculation'].range(self.MaxError).value
        TotalError = self.workbook.sheets['Calculation'].range(self.TotalError).value
        PVJitter = self.workbook.sheets['Calculation'].range(self.PVJitter).value
        AbsErrorQ2 = self.workbook.sheets['Calculation'].range(self.AbsErrorQ2).value
        AvgTrailError = self.workbook.sheets['Calculation'].range(self.trailingerror).value

        return [MaxError,TotalError,PVJitter,AbsErrorQ2,AvgTrailError,p,i]

    def getProcessValues(self):
        MaxInflow = self.workbook.sheets['Calculation'].range(self.MaxInflow).value
        MinLevel = self.workbook.sheets['Calculation'].range(self.MinLevel).value 
        MaxLevel = self.workbook.sheets['Calculation'].range(self.MaxLevel).value
        SetPoint = self.workbook.sheets['Calculation'].range(self.SetPoint).value 
        StaticInitialFout = self.workbook.sheets['Calculation'].range(self.StaticInitialFout).value
        Diameter = self.workbook.sheets['Calculation'].range(self.Diameter).value
        Section = self.workbook.sheets['Calculation'].range(self.Section).value 
        Disturbance = self.workbook.sheets['Calculation'].range(self.Disturbance).value
        DeadTime = self.workbook.sheets['Calculation'].range(self.DeadTime).value

        return SetPoint

    def getPID(self):

        p = self.workbook.sheets['Calculation'].range(self.Kp).value
        i = self.workbook.sheets['Calculation'].range(self.Ti).value
        d = self.workbook.sheets['Calculation'].range(self.Td).value

        return([p,i,d])







    

