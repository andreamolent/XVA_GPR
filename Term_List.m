function my_pol=Term_List(d,g)
if(d==0)
    my_pol=zeros(1,1);
else
if(g==0)
    my_pol=zeros(1,d);
else
   p1= Term_List(d,g-1);
   p2=p1(sum(p1,2)==(g-1),:);
   my_pol=p1;
   Id=eye(d);
   for l=1:size(p2,1)
    p3=p2(l,:)+Id;
    my_pol=[my_pol; p3];
   end
end
   my_pol=unique(my_pol,'row'); 
end
end