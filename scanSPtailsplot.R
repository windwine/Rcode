library(ggplot2)
for (delta in deltas)
  for (optionlife in optionlifes)
  {# optionlife = 40
    # delta=-0.05
   
    filename=paste0(abs(delta),"_",optionlife,"longvol.rds")
    ll<-readRDS(filename)
    colnames(ll)=c("Date","dailypnL")
    ll=ll %>% 
      mutate(AUM=cumsum(dailypnL))
    p1<-ll %>% 
      ggplot(aes(x=Date,y=AUM))+geom_line()+ggtitle(filename)
  }
    
  