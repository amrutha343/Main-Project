			fea = rand(7,10);
          options=[];
          options.ReducedDim=4;
			[eigvector,eigvalue] = PCA1(fea,options);
          Y = fea*eigvector;
          fe1=fea(1,:);
          Y1=fe1*eigvector