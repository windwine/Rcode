# library(RODBC)
library(quantmod)
graphics.off()
library(xts)
library(dplyr)
library(data.table)
require(PerformanceAnalytics)
library(rio)


rm(list=ls())

setwd("c:/eJiaqifiles")

load("adPrices2ETFadj.Rdata")

ids<-c(116070,106445,109820,122392, 212622, 116071, 150738,107899, 145999, 212622)
adPrice<-filter(adPrices, SecurityID %in% ids)

jump<-adPrice %>%
  filter(Adj!=lag(Adj))
temp<-filter(adPrices, SecurityID ==116070)

rm(adPrices)



stockprices<-readRDS("stockreturnsETF.rds")
# stockprices$Date<-as.integer((stockprices$Date))
stockprices$SecurityID<-as.integer((stockprices$SecurityID))
stockprices<-data.table(stockprices)
setkey(stockprices,Date, SecurityID)




# optiondata2<-readRDS("2005to2020ETFoptionsfewer.rds")
# optiondata2<-optiondata2 %>%
#   filter(Date>=as.Date("2009-01-01")) %>%
#   select(SecurityID,Date,Strike,Expiration,CallPut,BestBid,BestOffer,Delta,Vega,Gamma,Theta,OptionID,ImpliedVolatility)
# 
# saveRDS(optiondata2,"2005to2020ETFoptionsfewerlaptop.rds")

# optiondata2<-readRDS("2005to2020ETFoptionsfewer.rds")
optiondata2<-rio::import("2005to2020ETFoptionsfewer.fst")

# IDs=c(109820,106445,116908,110015,116070,125267,122392,126776,126681)
# 212622 VXX new 143580
IDs=c(109820)
num_ids=length(IDs)
results<-list()
port_stats=matrix(0,nrow=num_ids,ncol = 3)

for (ids in 1:num_ids)
{
  securityID=IDs[ids] #109820 SPY 110015 XLU 112873 IWR 125558 XHB 108105 SPX 116070 TLT 116069 LQD
  print(securityID)
  Adinfo<-filter(adPrice, SecurityID == securityID)
  Adinfo<-select(Adinfo,Date,splitAdj,Adj)
  Adinfo<-unique(Adinfo)
  
  optiondata<-filter(optiondata2,SecurityID==securityID)
  optiondata$Date=as.Date(optiondata$Date)
  optiondata$Expiration=as.Date(optiondata$Expiration)
  
  object.size(optiondata)/1e9
  # OptionID=32281736
  # 
  # optionret>-function(SecurityID,OptionID,startdate,enddate)
  # {
  #   options<-dpyr::filter(alloptions,OptionID==OptionID & Date>=startdate & Date<=enddate)
  # }
  # optiondata<-filter(optiondata,Date>=as.Date("2009-01-01"))
  optiondata$time2mat<-as.numeric(optiondata$Expiration-optiondata$Date)
  # OptionID1=123899769
  # OptionID2=123899770
  # startdate="2019-02-01"
  # enddate="2019-02-07"
  # SecurityID=161291
  
  optiondata<-data.table(optiondata)
  setkeyv(optiondata,c("Date","SecurityID"))
  stockprices<-data.table(stockprices)
  setkeyv(stockprices,c("Date","SecurityID"))
  
  alldates<-unique(optiondata$Date)
  
  if(tail(alldates,1)<as.Date("2019-12-1") || length(alldates)<1)
  {next;} # delisted ETF
  
  
  SKID=securityID
  SecurityID=SKID
  tempoptions=na.omit(optiondata[CJ(alldates,SKID),]) 
  
  fedfund<-read.csv("fedfund.csv")
  fedfund$Dates=as.Date(fedfund$Dates,format="%m/%d/%Y",origin="1970-01-01");
  colnames(fedfund)=c("Date","Fedrate")
  fedfund$Fedrate=fedfund$Fedrate/100/365
  
  # tempoptions=na.omit(optiondata[SecurityID==SKID,]) 
  # RBsignal<-read.csv("newsys3_5_mlr.csv",stringsAsFactors = F)
  RBsignal<-read.csv("newsys3_5_VVIX.csv",stringsAsFactors = F) # mainly as 1 for the short vol
  colnames(RBsignal)[1]<-"Date"
  # RBsignal$pos=ifelse(RBsignal$spread>0.05,-1,0)
  
  RBsignal<-RBsignal[,c("Date","pos")] # so you start the put when pos is -1.
  RBsignal$Date=as.Date(RBsignal$Date,origin="1970-01-01");
  
  RBsignal<-filter(RBsignal,Date>=min(optiondata$Date),Date<=max(optiondata$Date))
  # RBsignal$pos=-1*RBsignal$pos
  
  
  # tempoptions<-as.data.frame(tempoptions)
  hedgedretDT1<-function(SecurityID,startdate,enddate,OptionID1) #... are the optionIDs and the weights
  {
    OptionIDs<-c(OptionID1)
    # weights=c(1,0)
    # SecurityID=SKID
    
    days<-alldates[alldates>=startdate & alldates<=enddate]
    SKID=SecurityID
    
    # print((OptionIDs))
    # print(str(weights))
    # browser()
    
    options<-tempoptions[CJ(days,SKID),][OptionID %in% OptionIDs,] # the best choice
    
    options<-unique(options)
    # options$Date<-as.Date(options$Date)
    
    # system.time(dplyr::filter(stockprices,SecurityID==SKID & Date>=startdate & Date<=enddate))
    # system.time(stockprices[CJ(days,SKID),]) # the best choice
    stocks<-(stockprices[CJ(days,SKID),]) # the best choice
    
    #############optim only done to this part
    
    allinfo<-inner_join(options,stocks,by=c("Date","SecurityID"))
    allinfo<-left_join(allinfo,fedfund,by="Date") # a trick here.
    allinfo$Fedrate<-na.locf(allinfo$Fedrate,na.rm=FALSE, fromLast = T)
    allinfo$Fedrate<-na.locf(allinfo$Fedrate,na.rm=FALSE)
    allinfo$Fedrate[is.na(allinfo$Fedrate)]=0 #adjusted to reduce the data update 2020/11/28
    
    allinfo$midprice=(allinfo$BestBid+allinfo$BestOffer)/2
    allinfo$spread=(allinfo$BestOffer-allinfo$BestBid)
    allinfo$Date<-as.Date(allinfo$Date)
    
    allinfo<-left_join(allinfo,RBsignal,by="Date")
    allinfo<-inner_join(allinfo,Adinfo,by="Date")
    allinfo$pos[is.na(allinfo$pos)]=0 #adjusted to reduce the data update 2020/11/28
    
    allinfo<-arrange(allinfo,Date)
    allinfo$splitAdj<-allinfo$splitAdj/allinfo$splitAdj[1]
    allinfo$Price=allinfo$Price*allinfo$splitAdj
    allinfo$numcontract=allinfo$splitAdj
    
    sd(allinfo$ret[-1])*sqrt(252)
    days<-unique(allinfo$Date)
    obs=length(days)
    if (obs<=1)
    {
      finaldata<-c(1,0,1e9,1e9,1,1,1e9,1e9,0,1,0,securityID,0,0,0,0,0)
      # finaldata<-c(startcost,startspread,startvega,startprice,endcost1,endcost2,currentinfo$Strike/1000,currentinfo$Price,hedgeret,mat,endspread,securityID,interest1,interest2,interest3,RBexit,days[obs])
      
      portNAV<-NA
      portNAV2<-NA
      portNAV3<-NA
      
      allNAVs<-cbind(portNAV,portNAV2,portNAV3)
      ret_data=list(finaldata,allNAVs) # verified 2+3=1
      return(ret_data)
    }
    hedgeret=0
    interest1=0 # long bond and plus option delta
    interest2=0 # no money in the money market, only delta hedged delta
    interest3=0 # bond part, long bond
    
    oldinfo=dplyr::filter(allinfo,Date==days[1])
    startdelta=(oldinfo$Delta)[1]
    startcost=(oldinfo$midprice)[1]
    startspread=(oldinfo$spread)[1]
    startvega=(oldinfo$Vega)[1]
    startprice=unique(oldinfo$Price)
    
    
    #cal interest=start price and receive the interest (not compounding.)
    #buy options
    portNAV=rep(startprice,obs)-startspread*0.35*0 # buy the options, hold treasury, so interest part is stock-startcost
    portNAV2=rep(0,obs)-startspread*0.35*0 # buy the options,  so interest part is delta hedged stock-startcost
    portNAV3=rep(startprice,obs)-startspread*0.35*0 # hold the treasury
    
    deltamask=1
    
    for (i in 2:obs)
    {
      oldinfo=dplyr::filter(allinfo,Date==days[i-1])
      oldinfo$sign=ifelse(oldinfo$CallPut=="C",1,-1)
      oldinfo$Delta=ifelse(abs(oldinfo$Delta)>1,1*oldinfo$sign,oldinfo$Delta) # deal with very ITM data
      oldinfo<-oldinfo %>%
        arrange(desc(Delta))
      
      
      netdelta=(oldinfo$Delta)[1]*deltamask
      
      
      oldprice=unique(oldinfo$Price)
      oldtime2mat<-oldinfo$time2mat[1]
      
      currentinfo=dplyr::filter(allinfo,Date==days[i])
      if (nrow(currentinfo)<1)
      {
        next;
      }
      currentinfo$sign=ifelse(currentinfo$CallPut=="C",1,-1)
      currentinfo$Delta=ifelse(abs(currentinfo$Delta)>1,1*currentinfo$sign,currentinfo$Delta) # deal with very ITM data
      currentinfo<-currentinfo %>%
        arrange(desc(Delta))
      newtime2mat<-currentinfo$time2mat[1]
      daysdiff<-oldtime2mat-newtime2mat
      hedgeret=hedgeret-netdelta*oldprice*unique(currentinfo$ret)
      interest1=interest1+(startprice-startcost)*oldinfo$Fedrate[1]*daysdiff+netdelta*oldprice*oldinfo$Fedrate[1]*daysdiff # delta=-0.5, long 0.5 stock, pay 0.5 stock interest
      interest2=interest2+(-startcost)*oldinfo$Fedrate[1]*daysdiff+netdelta*oldprice*oldinfo$Fedrate[1]*daysdiff # delta=-0.5, long 0.5 stock, pay 0.5 stock interest
      interest3=interest3+(startprice)*oldinfo$Fedrate[1]*daysdiff
      
      portNAV[i]=startprice-startcost+interest1+(currentinfo$midprice[1])+hedgeret-startspread*0.35
      portNAV2[i]=-startcost+interest2*0+(currentinfo$midprice[1])+hedgeret-startspread*0.35
      portNAV3[i]=interest3+startprice-startspread*0.35
      
      if (i==obs && oldinfo$CallPut=="C")
      {
        exerciseprem=max(0,currentinfo$Price-currentinfo$Strike/1000)
      } else if (i==obs && oldinfo$CallPut=="P")
      {
        exerciseprem=max(0,currentinfo$Strike/1000-currentinfo$Price)
      }
      
      lastday=unique(as.Date(currentinfo$Expiration))
      if (! lastday %in% alldays)
      {
        lastday_index=max(which(alldates<=lastday))
        lastday=alldates[lastday_index]
      }
      mat<-days[obs]==lastday
      
      if (i==obs && mat==1)
      {
        portNAV[i]=startprice-startcost+interest1+(exerciseprem)+hedgeret-startspread*0.35*1 # account for the spread here
        portNAV2[i]=-startcost+interest2+(exerciseprem)+hedgeret-startspread*0.35*1
        
      } else if (i==obs && mat<1) 
      {
        portNAV[i]=startprice-startcost+interest1+((currentinfo$midprice)[1])+hedgeret-(currentinfo$spread)[1]*0.35*1-startspread*0.35*1
        portNAV2[i]=-startcost+interest2+((currentinfo$midprice)[1])+hedgeret-(currentinfo$spread)[1]*0.35*1-startspread*0.35*1
        # portNAV[i]=startprice-startcost+interest1+((currentinfo$BestBid)[1])+hedgeret
        # portNAV2[i]=-startcost+interest2+((currentinfo$BestBid)[1])+hedgeret
      }
      
      # print(days[i])
      # message=c(netdelta,hedgeret,interest1,interest2,interest3)
      # print(message)
      
      
    }
    
    endcost1=(currentinfo$midprice)[1]
    endcost2=unique(exerciseprem)
    endspread=(currentinfo$spread)[1]
    lastday=unique(as.Date(currentinfo$Expiration))
    if (! lastday %in% alldays)
    {
      lastday_index=max(which(alldates<=lastday))
      lastday=alldates[lastday_index]
    }
    mat<-days[obs]==lastday
    finaldata<-c(startcost,startspread,startvega,startprice,endcost1,endcost2,currentinfo$Strike/1000,currentinfo$Price,hedgeret,mat,endspread,SecurityID,interest1,interest2,interest3,startdelta)
    portNAV<-xts(portNAV,days)
    portNAV2<-xts(portNAV2,days)
    portNAV3<-xts(portNAV3,days)
    
    allNAVs<-cbind(portNAV,portNAV2,portNAV3)
    ret_data=list(finaldata,allNAVs) # verified 2+3=1
    return(ret_data)
    
    # write.csv(allinfo,"stradinfo1.csv")
  }
  
  # system.time(ll2<-hedgedretDT(SecurityID,startdate,enddate,OptionID1,OptionID2,1,-1))
  ptm<-Sys.time()
  alldays=sort(unique(tempoptions$Date))
  benchmark<-xts(rep(1,length(alldays)),as.Date(alldays))
  actiondays<-apply.daily(benchmark, tail, 1) #the last day in the trading interval
  actiondates=index(actiondays) # the dates we do the rebalance, we can change it to weekly
  obs=length(actiondates)
 
  
  
  
  RBsignal<-RBsignal %>% 
    mutate(pos=-1*pos)
  tradnum = 0
  entrydates = NULL #from RB table
  exitdates = NULL # from RB table
  if (RBsignal[1,"pos"]==-1) {entrydates=c(entrydates,(RBsignal[1,"Date"]));tradnum=tradnum+1}
  for (ii in 2:nrow(RBsignal))
  {
    if (RBsignal[ii,"pos"]==-1 && RBsignal[ii-1,"pos"]>=0) 
    {
      entrydates=c(entrydates,(RBsignal[ii,"Date"]))
      tradnum=tradnum+1
    } else if (RBsignal[ii-1,"pos"]==-1 && RBsignal[ii,"pos"]>=0) 
    {
      exitdates=c(exitdates,(RBsignal[ii,"Date"]))
    }
  }
  
  if (length(exitdates)<length(entrydates)) # deal with mismatch (notexited)
  {
    exitdates=c(exitdates,(RBsignal[nrow(RBsignal),"Date"]))
  }
  
  entrydates<-as.Date(entrydates)
  entrydates[47]="2024-08-02"
  exitdates<-as.Date(exitdates)
  sigdate<-data.frame(entrydates,exitdates)

  
  deltas=c(-0.05,-0.1)
  optionlifes=c(20,40)
  
  for (delta in deltas)
    for (optionlife in optionlifes)
    {# optionlife = 40
      # delta=-0.05
      ntrades=length(entrydates)
      count=1
      PnLs=NULL
      startdates<-NULL
      enddates<-NULL
      allstradinfo=list()
      allNAVs<-list()
      for (nn in 1:ntrades) #the last one is the current which we do not have any numbers
      {
        ii=which(actiondates==entrydates[nn])
        startdate=actiondates[ii]
        enddate=actiondates[min(ii+optionlife,length(actiondates))] # only hold it for 1 month then recycle
        
        # trade_num=which(entrydates==startdate)
        exitdate=exitdates[nn]
        
        time2exp<-as.numeric(enddate-startdate)
        
        tempbuyoptions=tempoptions[.(startdate,SKID),]
        # print(startdate)
        
        tempbuyoptions=tempbuyoptions[time2mat>=time2exp,]
        shorttest=min(tempbuyoptions$time2mat)
        tempbuyoptions=tempbuyoptions[time2mat==shorttest,]
        
        tempbuyoptions<-as.data.frame(tempbuyoptions)
        tempbuyoptions<-filter(tempbuyoptions,sign(Delta)==sign(delta))
        ATM_K<-tempbuyoptions %>%
          mutate(ddelta=abs(Delta-(delta)),min_ddelta=min(ddelta)) %>%
          filter(ddelta==min_ddelta)%>%
          select(Strike)
        ATM_K=as.numeric(ATM_K[[1]])[1] # added treatment
        
        Option1<-tempbuyoptions %>%
          filter(Strike==ATM_K) %>%
          filter(abs(Delta)<1) %>%  # the code is modified to be more robust for calls
          arrange(BestOffer-BestBid)
        
        if (dim(Option1)[1]!=1)
        {
          print(ii)
          next;
        }
        
        OptionID1=Option1$OptionID[1]
        enddate=Option1$Expiration # now hold until expiration
        print(paste("start a position for trade#",nn))
        print(Option1)
        
        while (exitdate>enddate)
        {
          ###PnL=rbind(PnL,TRADE(start,end))
          
          stradinfo=hedgedretDT1(securityID,startdate,enddate,OptionID1) 
          print(paste("finish a position for trade#",nn))
          
          # finaldata<-c(startcost,startspread,startvega,startprice,endcost1,endcost2,currentinfo$Strike/1000,currentinfo$Price,hedgeret,mat,endspread,SecurityID,interest1,interest2,interest3)
          ssinfo=stradinfo[[1]]
          allstradinfo[[count]]=ssinfo
          startdates<-c(startdates,startdate)
          enddates<-c(enddates,enddate)
          count=count+1
          pnlts<-stradinfo[[2]]
          sizecontrol=as.numeric(pnlts[1,3])
          sizecontrol=ssinfo[1]
          sizecontrol=Option1$Vega*Option1$ImpliedVolatility
          pnlts<-pnlts/sizecontrol
          # pnlts<-pnlts/sizecontrol*ssinfo[4]*0.02
          
          # startdelta=abs(ssinfo[16])
          # F=ssinfo[1]
          # startprice=ssinfo[4]
          # omega=startprice*startdelta/F
          # sizecontrol=1/F/omega
          # pnlts<-pnlts*sizecontrol
          # enddate=as.Date(tail(index(pnlts),1))
          
          pnlts<-apply(pnlts,2,diff)
          # pnlts<-pnlts[-1,]
          PnLs=rbind(PnLs,pnlts)
          
          
          new_ii=which.max(actiondates>=enddate) # to deal with 
          startdate=actiondates[new_ii]
          enddate=actiondates[min(new_ii+optionlife,length(actiondates))] # only hold it for 1 month then recycle
          
          
          time2exp<-as.numeric(enddate-startdate)
          
          tempbuyoptions=tempoptions[.(startdate,SKID),]
          # print(startdate)
          
          tempbuyoptions=tempbuyoptions[time2mat>=time2exp,]
          shorttest=min(tempbuyoptions$time2mat)
          tempbuyoptions=tempbuyoptions[time2mat==shorttest,]
          
          tempbuyoptions<-as.data.frame(tempbuyoptions)
          tempbuyoptions<-filter(tempbuyoptions,sign(Delta)==sign(delta))
          ATM_K<-tempbuyoptions %>%
            mutate(ddelta=abs(Delta-(delta)),min_ddelta=min(ddelta)) %>%
            filter(ddelta==min_ddelta)%>%
            select(Strike)
          ATM_K=as.numeric(ATM_K[[1]])[1] # added treatment
          
          Option1<-tempbuyoptions %>%
            filter(Strike==ATM_K) %>%
            filter(abs(Delta)<1) %>%  # the code is modified to be more robust for calls
            arrange(BestOffer-BestBid)
          
          if (dim(Option1)[1]!=1)
          {
            print(ii)
            next;
          }
          print(paste("start a new position for trade#",nn))
          print(Option1)
          OptionID1=Option1$OptionID[1]
          enddate=Option1$Expiration # now hold until expiration
          print(enddate)
          
        }
        enddate=exitdate
        if (enddate<=startdate)
        {next;}
        stradinfo=hedgedretDT1(securityID,startdate,enddate,OptionID1) 
        
        # finaldata<-c(startcost,startspread,startvega,startprice,endcost1,endcost2,currentinfo$Strike/1000,currentinfo$Price,hedgeret,mat,endspread,SecurityID,interest1,interest2,interest3)
        ssinfo=stradinfo[[1]]
        allstradinfo[[count]]=ssinfo
        startdates<-c(startdates,startdate)
        enddates<-c(enddates,enddate)
        count=count+1
        pnlts<-stradinfo[[2]]
        sizecontrol=as.numeric(pnlts[1,3])
        sizecontrol=ssinfo[1] # the startcost
        sizecontrol=Option1$Vega*Option1$ImpliedVolatility*1
        # sizecontrol=Option1$Vega
        pnlts<-pnlts/sizecontrol # so that is always $1 to buy
        # pnlts<-pnlts/sizecontrol*ssinfo[4]*0.02
        
        # startdelta=abs(ssinfo[16])
        # F=ssinfo[1]
        # startprice=ssinfo[4]
        # omega=startprice*startdelta/F
        # sizecontrol=1/F/omega
        # pnlts<-pnlts*sizecontrol
        
        
        if (nrow(pnlts)>2)
        {
          pnlts<-apply(pnlts,2,diff)
        } else
        {
          pnlts=pnlts[2,]-as.numeric(pnlts[1,])
        }
        
        
        # pnlts<-pnlts[-1,]
        PnLs=rbind(PnLs,pnlts)
        # allstradinfo=rbind(allstradinfo,stradinfo[[1]])
        
        
      }
      Sys.time()-ptm
      dates<-as.Date(time(PnLs))
      PnLs<-xts(coredata(PnLs),order.by = dates)
      
      # index(PnLs)<-as.Date(index(PnLs))
      
      fake=xts(rep(1,length(actiondates)),order.by = actiondates)
      final<-merge.xts(fake,PnLs[,2],join="outer")
      notrade<-is.na(as.numeric(final[,2]))
      final[notrade,2]<-0
      final<-final["2008::"]
      op = function(x, d=3) sprintf(paste0("%1.",d,"f"), x) 
      print(plot(cumsum(final[,2]),main=paste(SKID, "Delta=", delta,"Optionlife=", optionlife)))
      
      ll=(final[,2])
      ll=data.frame(index(ll),coredata(ll))
      filename=paste0(abs(delta),"_",optionlife,"longvol.rds")
      saveRDS(ll,filename)
      
      xx<-cumsum(PnLs[,2])
      allstradinfo<-data.frame(Reduce(rbind, allstradinfo))#unlist(allstradinfo)
      colnames(allstradinfo)=c("startcost","startspread","startvega","startprice","endcost1","endcost2","K","endprice","hedgeret","mat","endspread","SecuirtyID","interest1","interest2","interest3","startdelta")  
      
      allstradinfo$endcost=ifelse(allstradinfo$mat==0,allstradinfo$endcost1-allstradinfo$endspread*0.35,allstradinfo$endcost2)
      
      allstradinfo$PnL=allstradinfo$startcost*(-1)+allstradinfo$endcost*(1)+allstradinfo$hedgeret*(1)+allstradinfo$interest2*1-0.35*allstradinfo$startspread
      
      allstradinfo$startdates<-as.Date(startdates)
      allstradinfo$enddates<-as.Date(enddates)
      
      filename=paste0(abs(delta),"_",optionlife,"longvolsummary.csv")
      write.csv(allstradinfo,filename)}
  
}
