# 深入研究
完成第一次使用超級電腦，接下來更深入了解超級電腦的進階使用。
## 資料傳輸 
[晶創25](https://man.twcc.ai/Yg_dk6n2T_-Y2Pmx3fXzAA)   
[創進一號](https://man.twcc.ai/@f1-manual/transport_ip)   
[台灣杉三號](https://man.twcc.ai/@TWCC-III-manual/SyGsFqRSt)  
[台灣杉二號](https://man.twcc.ai/@twccdocs/doc-hfs-main-zh/%2F%40twccdocs%2Fguide-hfs-connect-to-data-transfer-node-zh)  
## 圖形介面登入  
[晶創25](https://man.twcc.ai/B5MmH5TiS6qwCKYZnP7HBw)   
[創進一號](https://man.twcc.ai/@f1-manual/thinlinc)  
[台灣杉三號](https://man.twcc.ai/@TWCC-III-manual/r1_eaoM2u)    
## Open OnDemand  
[創進一號:5.特殊工作節點操作流程](https://man.twcc.ai/@f1-manual/manual)   
## 查詢資源
查詢Partition狀態，以晶創25為例  
```shell
[<account>@cbi-lgn01 ~]$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
dev          up    2:00:00     10    mix hgpn[02-06,17-21]
normal       up 2-00:00:00     10    mix hgpn[02-06,17-21]
4nodes       up 1-00:00:00     10    mix hgpn[02-06,17-21]
normal2      up 2-00:00:00      8    mix hgpn[39-46]
```   

查詢節點有閒置的GPU，以下是晶創25有閒置GPU的節點  
```shell
[<account>@cbi-lgn01 ~]$ scontrol show node | grep -E 'NodeName|AllocTRES' | sed -n '/NodeName/{N; /gres\/gpu=8/!p}'
NodeName=hgpn02 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=32,mem=1400G,gres/gpu=7
NodeName=hgpn03 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=24,mem=800G,gres/gpu=4
NodeName=hgpn04 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=14,mem=1000G,gres/gpu=5
NodeName=hgpn06 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=13,mem=800G,gres/gpu=4
NodeName=hgpn19 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=9,mem=1200G,gres/gpu=6
NodeName=hgpn39 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=5,mem=200G,gres/gpu=1
NodeName=hgpn43 Arch=x86_64 CoresPerSocket=56
   AllocTRES=cpu=84,mem=1400G,gres/gpu=7
```  
## 容器打包
Singularity...


