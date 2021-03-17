function collected_data = collect(data1, data2, data3)
    [~,I] = min(data1(1:20,2));
    data1 = data1(I:end,:);
    [~,I] = min(data2(1:20,2));
    data2 = data2(I:end,:);
    [~,I] = min(data3(1:20,2));
    data3 = data3(I:end,:);
    data2 = data2(1:length(data1), :);
    data3 = data3(1:length(data1), :);
    collected_data = [data1'; data2'; data3'];

end