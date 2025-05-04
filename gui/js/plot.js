function drawAxis() {
    let cvs = document.getElementById("plot") 
    let ctx = cvs.getContext("2d");
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;

    let y0 =  0;
    let y1 = cvs.height - 25;
    let x0 = 25;
    let x1 = cvs.width - 10;

    ctx.lineWidth = 2;
    ctx.moveTo( x0, y0);
    ctx.lineTo( x1, y0);
    ctx.lineTo( x1, y1);
    ctx.lineTo( x0, y1);
    ctx.lineTo( x0, y0);

    ctx.stroke()

    ctx.fillStyle = "white";
    ctx.lineWidth = 1;
    ctx.font = "20px Times New Roman";
    for (k = 0; k <= 4; k++) {
        ctx.fillText("0", (k*x0 + (4-k)*x1)/4 - 5, y1 + 20);
    }


    ctx.setLineDash([2,2]);  

    ctx.moveTo(x0, (y0 + y1)/2);
    ctx.lineTo(x1, (y0 + y1)/2);

    for (k = 1; k < 4; k++) {
        ctx.moveTo((k*x0 + (4-k)*x1)/4, y0);
        ctx.lineTo((k*x0 + (4-k)*x1)/4, y1);
    }

    ctx.stroke();

}