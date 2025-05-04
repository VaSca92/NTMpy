const colors = ['#a88924','#2469a8','#a84e24','#44911a','#6319af'];
let layers;


async function drawMaterial_core(labels) {
    
    $(".canvas div").remove();

    const style1 = ' style="text-align:center;color:#000000;height:30px; font-size:24">';
    const style2 = ' style="width:100%; height: 20px; background-color:'

    layers = await eel.getLayers()()

    for (let i = 0; i < layers.length; i++) {
        $(".canvas").append('<div style="flex:' + layers[i].length + '">' + 
                                '<div' + style1 + labels[i] + '</div>' +
                                '<div' + style2 + colors[i%5] + '"></div>' + 
                            '<div>');
    }

    return layers.length;
} 