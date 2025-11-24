// 使用 Material for MkDocs 專屬的訂閱事件
// 這能確保即使開啟了 "navigation.instant" (SPA模式)，腳本也能正常運作
document$.subscribe(function() {
  var links = document.querySelectorAll('a[href^="http"]');
  
  links.forEach(function(link) {
    // 檢查連結是否指向外部網站 (與當前網站網域不同)
    if (link.hostname !== window.location.hostname) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
      // 可選：加上一個小圖示類別，如果你想要特別標記外部連結的話
      link.classList.add('external-link'); 
    }
  });
});