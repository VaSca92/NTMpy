function drawAxis() {
    let canvas = document.getElementById("plot") 
    let ctx = canvas.getContext("2d");
    ctx.strokeStyle = "white";

    let y0 =  0;
    let y1 = canvas.height - 25;
    let x0 = 25;
    let x1 = canvas.width - 10;

    ctx.lineWidth = 2;
    ctx.setLineDash([2,0]); 

    ctx.beginPath();
    ctx.moveTo( x0, y0);
    ctx.lineTo( x1, y0);
    ctx.lineTo( x1, y1);
    ctx.lineTo( x0, y1);
    ctx.lineTo( x0, y0);

    ctx.stroke()

    /*ctx.fillStyle = "white";
    ctx.lineWidth = 1;
    ctx.font = "20px Times New Roman";
    for (k = 0; k <= 4; k++) {
        ctx.fillText("0", (k*x0 + (4-k)*x1)/4 - 5, y1 + 20);
    }*/

    ctx.lineWidth = 1;
    ctx.setLineDash([2,4]);  

    ctx.beginPath();
    ctx.moveTo(x0, (y0 + y1)/2);
    ctx.lineTo(x1, (y0 + y1)/2);

    for (k = 1; k < 4; k++) {
        ctx.moveTo((k*x0 + (4-k)*x1)/4, y0);
        ctx.lineTo((k*x0 + (4-k)*x1)/4, y1);
    }

    ctx.stroke();

}

function drawCurve(data) {
    let canvas = document.getElementById("plot"); 
    let ctx = canvas.getContext("2d");
    ctx.strokeStyle = "white";

    let y0 =  0;
    let y1 = canvas.height - 25;
    let x0 = 25;
    let x1 = canvas.width - 10;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxis();
        
    const Ymax = 1.2 * Math.max(...data);
    const Xmax = data.length - 1;
    
    // Plot data
    ctx.setLineDash([2,0]);
    ctx.strokeStyle = '#ccccff';
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((ydata, xdata) => {
        const x = x0 + (xdata/Xmax) * (x1-x0);
        const y = y1 - (ydata/Ymax) * (y1-y0);
        xdata === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });

    ctx.stroke();
}