function onClick(event){
    var x = event.offsetX;
    var y = event.offsetY;
    var cords = {y_cord : y, x_cord : x};
    cords = JSON.stringify(cords);
    document.getElementById("cords").value = cords
    document.getElementById("demo").innerHTML= cords
}
