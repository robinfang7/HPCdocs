# 資料傳輸 
[晶創25](https://man.twcc.ai/Yg_dk6n2T_-Y2Pmx3fXzAA)   
[創進一號](https://man.twcc.ai/@f1-manual/transport_ip)   
[台灣杉三號](https://man.twcc.ai/@TWCC-III-manual/SyGsFqRSt)  
[台灣杉二號](https://man.twcc.ai/@twccdocs/doc-hfs-main-zh/%2F%40twccdocs%2Fguide-hfs-connect-to-data-transfer-node-zh)  
# 圖形介面登入  
[晶創25](https://man.twcc.ai/B5MmH5TiS6qwCKYZnP7HBw)   
[創進一號](https://man.twcc.ai/@f1-manual/thinlinc)  
[台灣杉三號](https://man.twcc.ai/@TWCC-III-manual/r1_eaoM2u)    
# Open OnDemand  
[創進一號:特殊工作節點操作流程](https://man.twcc.ai/@f1-manual/manual)   
# 查詢資源
查詢節點狀態  
`sinfo`  
查詢節點有空的GPU  
```shell
scontrol show node | grep -E 'NodeName|AllocTRES' | sed -n '/NodeName/{N; /gres\/gpu=8/!p}'
```  
# 容器打包
Singularity...


