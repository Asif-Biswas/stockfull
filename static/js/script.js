// Menu Toogler

let sideNav= document.getElementById('sidenav');
let toggler=document.getElementById('toggle__menu')

toggler.addEventListener('click',function(e) {
    sideNav.classList.toggle('show');
})
